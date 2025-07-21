# This file extracted from drozpack (ablejones stuff) 2025-07-19
# 2025-07-21: Added WanVacePhantomToVideo node
# sorry if it doesn't work!

import torch
import logging
import comfy.ldm.wan.model
from comfy.ldm.wan.model import sinusoidal_embedding_1d
import nodes
import node_helpers

def wrap_vace_phantom_wan_model(model):
    """
    Patch a VaceWanModel to support per-frame strength and Phantom embeds.
    Based on the WanVideoTeaCacheKJ implementation in comfyui-kjnodes.
    """
    model_clone = model.clone()
    
    # Check if this is a VaceWanModel
    diffusion_model = model_clone.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, comfy.ldm.wan.model.VaceWanModel):
        logging.info("Not a VaceWanModel, skipping per-frame strength patch")
        return model_clone
    
    def outer_wrapper():
        def unet_wrapper_function(model_function, kwargs):
            # Extract parameters from kwargs
            input_data = kwargs["input"]
            timestep = kwargs["timestep"] 
            c = kwargs["c"]
            
            vace_strength = c.get("vace_strength", [1.0])

            # Check if we have nested lists (our separate reference strength format)
            use_patched_forward_orig = (
                isinstance(vace_strength, list) and 
                len(vace_strength) > 0 and
                isinstance(vace_strength[0], list)
            )

            # Also check if we have Phantom concats (c.get("time_dim_concat") exists and is a tensor)
            use_patched_forward_orig = use_patched_forward_orig or (
                c.get("time_dim_concat") is not None and
                isinstance(c.get("time_dim_concat"), torch.Tensor)
            )

            if use_patched_forward_orig:
                # Use the patched forward_orig method
                from unittest.mock import patch
                
                forward_function = _vaceph_forward_orig
                context = patch.multiple(
                    diffusion_model,
                    forward_orig=forward_function.__get__(diffusion_model, diffusion_model.__class__)
                )
                
                with context:
                    out = model_function(input_data, timestep, **c)
            else:
                # Use original behavior
                out = model_function(input_data, timestep, **c)
            
            return out
        return unet_wrapper_function
    
    model_clone.set_model_unet_function_wrapper(outer_wrapper())
    
    logging.info("VaceWanModel patched for separate reference strength support using unet wrapper")
    return model_clone


def unwrap_vace_phantom_wan_model(model):
    """
    Remove the patch from a VaceWanModel.
    """
    model_clone = model.clone()
    logging.info("VaceWanModel patch removed")
    return model_clone

def _vaceph_forward_orig(
    self,
    x,
    t,
    context,
    vace_context,
    vace_strength,
    vace_has_reference=False,
    clip_fea=None,
    freqs=None,
    transformer_options={},
    **kwargs,
):
    """
    Custom forward implementation that applies separate strength values for reference and control content.
    Also handles Phantom Subject compatibility by ensuring VACE frames match the expected sequence length.
    This is adapted from the original VaceWanModel.forward_orig method.
    """
    
    # embeddings
    x = self.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)

    # time embeddings
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x.dtype))
    e0 = self.time_projection(e).unflatten(1, (6, self.dim))

    # context
    context = self.text_embedding(context)

    context_img_len = None
    if clip_fea is not None:
        if self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)
        context_img_len = clip_fea.shape[-2]

    # Process vace_context
    orig_shape = list(vace_context.shape)
    vace_context = vace_context.movedim(0, 1).reshape([-1] + orig_shape[2:])
    c = self.vace_patch_embedding(vace_context.float()).to(vace_context.dtype)
    vace_grid_sizes = c.shape[2:]  # Get VACE patch dimensions (t_patches, h_patches, w_patches)
    c = c.flatten(2).transpose(1, 2)
    
    # Store expected sequence length from x for comparison
    expected_seq_length = x.shape[1]
    
    # Split into batch segments
    c = list(c.split(orig_shape[0], dim=0))
    
    # arguments
    x_orig = x

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    
    for i, block in enumerate(self.blocks):
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len)
                return out
            out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap})
            x = out["img"]
        else:
            x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)

        ii = self.vace_layers_mapping.get(i, None)
        if ii is not None:
            for iii in range(len(c)):
                # Check and fix sequence length mismatch before passing to vace_blocks
                if c[iii].shape[1] < expected_seq_length:
                    # Calculate how many frames to add
                    seq_length_per_frame = grid_sizes[1] * grid_sizes[2]
                    extra_seq_length = expected_seq_length - c[iii].shape[1]
                    extra_frames = extra_seq_length // seq_length_per_frame
                    # Create a tensor of shape [1, 16, extra_frames, orig_shape[-2], orig_shape[-1]]
                    # Keep new tensors on CPU until added to the sequence
                    SCALE_FACTOR = 0.5  # Scale factor (what should this be?!)
                    solid_frames = torch.ones(1, 16, extra_frames, orig_shape[-2], orig_shape[-1], device="cpu", dtype=c[iii].dtype) * SCALE_FACTOR

                    from comfy.latent_formats import Wan21
                    processed_frames = Wan21().process_out(solid_frames)
                    processed_frames = self.patch_embedding(processed_frames.float()).to(processed_frames.dtype)
                    c_padding = processed_frames.flatten(2).transpose(1, 2)

                    # Concatenate along sequence dimension to extend c
                    c[iii] = torch.cat([c[iii], c_padding.to(c[iii].device)], dim=1)
                    
                    # if i == 0 and iii == 0:
                    #     logging.info(f"\nVACE-Phantom compatibility: Extended VACE frames by adding {extra_frames} frames to match sequence length {expected_seq_length}.\n")

                    del processed_frames, c_padding, solid_frames

                elif c[iii].shape[1] > expected_seq_length:
                    # If c[iii] is longer than expected, truncate it
                    logging.warning(f"Truncating VACE frames for batch {iii} from {c[iii].shape[1]} to {expected_seq_length} frames.")
                    c[iii] = c[iii][:, :expected_seq_length, :]

                # Continue with original processing
                c_skip, c[iii] = self.vace_blocks[ii](
                    c[iii], x=x_orig, e=e0, freqs=freqs, 
                    context=context, context_img_len=context_img_len
                )
                
                # CUSTOM LOGIC: Apply separate reference strength if provided
                # iii is the batch index, vace_strength[iii] is the strength for this batch item
                if iii < len(vace_strength):
                    batch_strength = vace_strength[iii]
                else:
                    # Fallback to last strength if list is shorter
                    batch_strength = vace_strength[-1]
                
                # Handle nested list (separate reference/control strengths) or single value
                if isinstance(batch_strength, list):
                    # We have separate strengths for reference and control
                    # Assume 1 reference frame when separate strengths are provided
                    reference_frames = 1
                    x = _apply_separate_reference_strength(x, c_skip, batch_strength, reference_frames, vace_grid_sizes)
                else:
                    # Single strength value - original behavior
                    x += c_skip * batch_strength
                
            del c_skip

    # head
    x = self.head(x, e)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return x


def _apply_separate_reference_strength(x, c_skip, strength_list, reference_frames=0, vace_grid_sizes=None):
    """
    Apply separate strength values for reference and control content.
    
    The VACE model processes reference and control frames together in a concatenated form.
    The reference frame is at the beginning of the temporal dimension.
    
    Args:
        x: Main feature tensor [B, sequence_length, hidden_dim]
        c_skip: VACE skip connection [B, sequence_length, hidden_dim]
        strength_list: List with [reference_strength, control_strength]
        reference_frames: Number of reference frames (0 if no reference)
        vace_grid_sizes: Actual patch grid sizes (t_patches, h_patches, w_patches) from VACE embedding
    """
    
    # Extract strengths
    if len(strength_list) >= 2:
        reference_strength = strength_list[0]
        control_strength = strength_list[1]
    else:
        # Fallback to single strength
        reference_strength = control_strength = strength_list[0]
    
    # If no reference frames, apply control strength uniformly
    if reference_frames == 0:
        weighted_c_skip = c_skip * control_strength
        x += weighted_c_skip
        return x
    
    # Calculate reference portion size in the sequence dimension
    # The reference frames are at the beginning of the temporal dimension
    # After patch embedding: sequence_length = t_patches * h_patches * w_patches
    # Reference portion = reference_frames * h_patches * w_patches
    
    # Get tensor dimensions
    batch_size, sequence_length, hidden_dim = c_skip.shape
    
    try:
        if vace_grid_sizes is not None and len(vace_grid_sizes) >= 3:
            # Use actual patch dimensions from VACE embedding
            t_patches, h_patches, w_patches = vace_grid_sizes
            
            # Calculate patches per frame (h_patches * w_patches)
            patches_per_frame = h_patches * w_patches
            
            # Calculate reference sequence length
            reference_sequence_length = reference_frames * patches_per_frame
            
            # Clamp to valid range
            reference_sequence_length = max(0, min(reference_sequence_length, sequence_length))
            
            # logging.info(f"VACE reference strength split: reference_frames={reference_frames}, "
            #             f"patches_per_frame={patches_per_frame}, reference_seq_len={reference_sequence_length}, "
            #             f"total_seq_len={sequence_length}")
            
        else:
            # Fallback: assume reference takes up proportional space
            reference_sequence_length = (reference_frames * sequence_length) // max(1, reference_frames + 1)
            reference_sequence_length = max(0, min(reference_sequence_length, sequence_length))
            
            logging.warning(f"VACE grid sizes not available, using proportional split: {reference_sequence_length}/{sequence_length}")
        
        if reference_sequence_length > 0:
            # Split the c_skip tensor
            c_skip_reference = c_skip[:, :reference_sequence_length, :]
            c_skip_control = c_skip[:, reference_sequence_length:, :]
            
            # Apply separate strengths
            weighted_reference = c_skip_reference * reference_strength
            weighted_control = c_skip_control * control_strength
            
            # Concatenate back together
            weighted_c_skip = torch.cat([weighted_reference, weighted_control], dim=1)
            
            # logging.info(f"Applied separate strengths: ref={reference_strength}, ctrl={control_strength}")
        else:
            # If calculation failed, use control strength
            weighted_c_skip = c_skip * control_strength
            logging.warning("Reference sequence length is 0, using control strength")
            
    except Exception as e:
        # Fallback to control strength if any calculation fails
        logging.warning(f"Failed to calculate reference sequence split, using control strength: {e}")
        weighted_c_skip = c_skip * control_strength
    
    # Add to main features
    x += weighted_c_skip
    
    return x

class VacePhantomWanModelPatcher:
    """
    Node to enable per-frame strength support for VacePhantomWanModel.
    This patches the model to support separate strength values for reference and control frames.
    """
    def __init__(self) -> None:
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enable": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_model"
    CATEGORY = "drozpack/experimental"
    
    def patch_model(self, model, enable):
        """
        Wrap or unwrap the VaceWanModel to enable/disable per-frame strength support.
        
        Args:
            model: The model to patch
            enable: Whether to enable or disable per-frame strength support and Phantom compatibility
            
        Returns:
            The patched model
        """
        # Clone the model to avoid modifying the original
        m = model.clone()
        
        if enable:
            m = wrap_vace_phantom_wan_model(m)
        else:
            m = unwrap_vace_phantom_wan_model(m)

        logging.info(f"VacePhantomModel patch {'enabled' if enable else 'disabled'}")
        return (m,)
    
class WanVaceToVideoAdvanced:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                             "strength_reference": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                },
                "optional": {"control_video": ("IMAGE", ),
                             "control_masks": ("MASK", ),
                             "reference_image": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "trim_latent")
    FUNCTION = "encode"

    CATEGORY = "drozpack/experimental"

    EXPERIMENTAL = True

    def encode(self, positive, negative, vae, width, height, length, batch_size, strength, strength_reference, control_video=None, control_masks=None, reference_image=None):
        latent_length = ((length - 1) // 4) + 1
        if control_video is not None:
            control_video = comfy.utils.common_upscale(control_video[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            if control_video.shape[0] < length:
                control_video = torch.nn.functional.pad(control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), value=0.5)
        else:
            control_video = torch.ones((length, height, width, 3)) * 0.5

        if reference_image is not None:
            reference_image = comfy.utils.common_upscale(reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            reference_image = vae.encode(reference_image[:, :, :, :3])
            reference_image = torch.cat([reference_image, comfy.latent_formats.Wan21().process_out(torch.zeros_like(reference_image))], dim=1)
            # Create nested list only if strengths are different
            if strength_reference != strength:
                strength_list = [[strength_reference, strength]] * batch_size
            else:
                strength_list = [strength] * batch_size
        else:
            strength_list = [strength] * batch_size

        if control_masks is None:
            mask = torch.ones((length, height, width, 1))
        else:
            mask = control_masks
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = comfy.utils.common_upscale(mask[:length], width, height, "bilinear", "center").movedim(1, -1)
            if mask.shape[0] < length:
                mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]), value=1.0)

        control_video = control_video - 0.5
        inactive = (control_video * (1 - mask)) + 0.5
        reactive = (control_video * mask) + 0.5

        inactive = vae.encode(inactive[:, :, :, :3])
        reactive = vae.encode(reactive[:, :, :, :3])
        control_video_latent = torch.cat((inactive, reactive), dim=1)
        if reference_image is not None:
            control_video_latent = torch.cat((reference_image, control_video_latent), dim=2)

        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
        mask = mask.permute(2, 4, 0, 1, 3)
        mask = mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode='nearest-exact').squeeze(0)

        trim_latent = 0
        if reference_image is not None:
            mask_pad = torch.zeros_like(mask[:, :reference_image.shape[2], :, :])
            mask = torch.cat((mask_pad, mask), dim=1)
            latent_length += reference_image.shape[2]
            trim_latent = reference_image.shape[2]

        mask = mask.unsqueeze(0)

        positive = node_helpers.conditioning_set_values(positive, {
            "vace_frames": [control_video_latent],
            "vace_mask": [mask],
            "vace_strength": strength_list,
        }, append=True)
        negative = node_helpers.conditioning_set_values(negative, {
            "vace_frames": [control_video_latent],
            "vace_mask": [mask],
            "vace_strength": strength_list,
        }, append=True)

        latent = torch.zeros([batch_size, 16, latent_length, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent, trim_latent)
    


class WanVacePhantomToVideo:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                },
                "optional": {"control_video": ("IMAGE", ),
                             "control_masks": ("MASK", ),
                             "vace_references": ("IMAGE", ),
                             "phantom_images": ("IMAGE", ),
                             "phantom_mask_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Vace mask value for the Phantom embed region."}),
                             "phantom_control_value": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Padded vace embedded latents value for the Phantom embed region."}),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "neg_phant_img", "latent", "trim_latent")
    FUNCTION = "encode"

    CATEGORY = "drozpack/experimental"

    EXPERIMENTAL = True

    def encode(self, positive, negative, vae, width, height, length, batch_size, strength, 
               control_video=None, control_masks=None, vace_references=None, phantom_images=None,
               phantom_mask_value=0.0, phantom_control_value=0.5):
        
        # Get the number of images in the phantom_images
        if phantom_images is not None:
            num_phantom_images = min(length, phantom_images.shape[0])
        else:
            num_phantom_images = 0

        # Calculate the additional length needed for Phantom images in Vace embeds
        phantom_padding = num_phantom_images * 4
        vace_length = length + phantom_padding

        ########################
        # execute WanVaceToVideo logic
        vace_latent_length = ((vace_length - 1) // 4) + 1
        if control_video is not None:
            control_video = comfy.utils.common_upscale(control_video[:vace_length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            if control_video.shape[0] < vace_length:
                control_video = torch.nn.functional.pad(control_video, (0, 0, 0, 0, 0, 0, 0, vace_length - control_video.shape[0]), value=0.5)
        else:
            control_video = torch.ones((vace_length, height, width, 3)) * 0.5

        if vace_references is not None:
            vace_references = comfy.utils.common_upscale(vace_references[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            vace_references = vae.encode(vace_references[:, :, :, :3])
            vace_references = torch.cat([vace_references, comfy.latent_formats.Wan21().process_out(torch.zeros_like(vace_references))], dim=1)

        if control_masks is None:
            mask = torch.ones((vace_length, height, width, 1))
        else:
            mask = control_masks
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = comfy.utils.common_upscale(mask[:vace_length], width, height, "bilinear", "center").movedim(1, -1)
            if mask.shape[0] < vace_length:
                mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, 0, 0, vace_length - mask.shape[0]), value=1.0)

        #  modified mask for VACE processing
        mask_modified = mask.clone()
        
        # Modify the phantom-overlapping portions if phantom images exist
        if phantom_padding > 0:
            # modify mask values for phantom padding region
            mask_modified[length:, :, :, :] = phantom_mask_value
            # modify control video values for phantom padding region
            control_video[length:, :, :, :] = phantom_control_value
        
        control_video = control_video - 0.5
        inactive = (control_video * (1 - mask_modified)) + 0.5
        reactive = (control_video * mask_modified) + 0.5

        inactive = vae.encode(inactive[:, :, :, :3])
        reactive = vae.encode(reactive[:, :, :, :3])
        control_video_latent = torch.cat((inactive, reactive), dim=1)
        if vace_references is not None:
            control_video_latent = torch.cat((vace_references, control_video_latent), dim=2)

        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        mask_modified = mask_modified.view(vace_length, height_mask, vae_stride, width_mask, vae_stride)
        mask_modified = mask_modified.permute(2, 4, 0, 1, 3)
        mask_modified = mask_modified.reshape(vae_stride * vae_stride, vace_length, height_mask, width_mask)
        mask_modified = torch.nn.functional.interpolate(mask_modified.unsqueeze(0), size=(vace_latent_length, height_mask, width_mask), mode='nearest-exact').squeeze(0)

        trim_latent = 0
        if vace_references is not None:
            mask_pad = torch.zeros_like(mask_modified[:, :vace_references.shape[2], :, :])
            mask_modified = torch.cat((mask_pad, mask_modified), dim=1)
            vace_latent_length += vace_references.shape[2]
            trim_latent = vace_references.shape[2]

        mask_modified = mask_modified.unsqueeze(0)

        positive = node_helpers.conditioning_set_values(positive, {"vace_frames": [control_video_latent], "vace_mask": [mask_modified], "vace_strength": [strength]}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"vace_frames": [control_video_latent], "vace_mask": [mask_modified], "vace_strength": [strength]}, append=True)

        ######################
        # execute WanPhantomSubjectToVideo logic
        phantom_length = length
        # Create the latent from WanPhantomSubjectToVideo (this will be our output latent)
        latent = torch.zeros([batch_size, 16, ((phantom_length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        
        # WanPhantomSubjectToVideo uses the negative from WanVace as input and creates two outputs
        neg_phant_img = negative  # This becomes negative_img_text (with zeros)
        
        if phantom_images is not None:
            phantom_images = comfy.utils.common_upscale(phantom_images[:phantom_length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            latent_images = []
            for i in phantom_images:
                latent_images += [vae.encode(i.unsqueeze(0)[:, :, :, :3])]
            concat_latent_image = torch.cat(latent_images, dim=2)

            # Apply phantom logic: positive gets the phantom images
            positive = node_helpers.conditioning_set_values(positive, {"time_dim_concat": concat_latent_image})
            # negative_text (cond2 in original) gets the phantom images 
            negative = node_helpers.conditioning_set_values(negative, {"time_dim_concat": concat_latent_image})
            # neg_phant_img (negative in original) gets zeros instead of phantom images
            neg_phant_img = node_helpers.conditioning_set_values(negative, {"time_dim_concat": comfy.latent_formats.Wan21().process_out(torch.zeros_like(concat_latent_image))})

        # Adjust latent size if reference image was provided (matching WanVaceToVideo logic)
        if vace_references is not None:
            # Prepend zeros to match the reference image dimensions
            latent_pad = torch.zeros([batch_size, 16, vace_references.shape[2], height // 8, width // 8], device=latent.device, dtype=latent.dtype)
            latent = torch.cat([latent_pad, latent], dim=2)
        
        out_latent = {}
        out_latent["samples"] = latent
        
        return (positive, negative, neg_phant_img, out_latent, trim_latent)


NODE_CLASS_MAPPINGS = {
    "WanVacePhantomToVideo": WanVacePhantomToVideo,
    "VacePhantomWanModelPatcher": VacePhantomWanModelPatcher,
    "WanVaceToVideoAdvanced": WanVaceToVideoAdvanced,
}