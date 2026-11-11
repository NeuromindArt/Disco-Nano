"""Every setting the renderer accepts, with its default and help text."""
from models import DIFFUSION, MAIN as MAIN_MODEL


def field(name, default, section, help_text, kind=None, options=None):
    if kind is None:
        kind = "boolean" if isinstance(default, bool) else "number" if isinstance(default, (int, float)) else "string"
    return {"name": name, "default": default, "section": section, "help": help_text, "kind": kind, "options": options}


MAIN = "Main"
GUIDANCE = "Guidance"
CUTS = "Cutouts"
SAMPLING = "Sampling"
CLIP_MODELS = "More CLIP models"
SYMMETRY = "Symmetry"
TILES = "Tiles"
LAB = "Experimental"
CONTROLS = "Controls"
HANDOFF = "Handoff"
ANIMATION = "Animation"

FIELDS = [
    field("steps", 200, MAIN, "How many steps the image is refined over."),
    field("skip_steps", 0, MAIN, "{How much of the picture is painted. 100% paints from pure noise. With a start image, about 50% keeps its layout and colors and repaints the details, and lower stays closer to it. With text only, keep it at 100%: anything lower throws away the steps that lay out the scene.|Steps skipped at the start of the render. With a start image, about half of steps keeps its layout and colors and repaints the details, and more stays closer to it. With text only, keep it at 0: skipping throws away the steps that lay out the scene.}"),
    field("clip_guidance_scale", 10000, MAIN, "How strongly the image is pushed toward the prompt."),
    field("n_batches", 1, MAIN, "How many images to render."),
    field("set_seed", "random_seed", MAIN, "A number starts the render from the same noise. Turn on {exact_repeat} for an identical image."),
    field("exact_repeat", False, MAIN, "The same seed and settings give the identical image on this PC. About 30% slower."),
    field("cpu_threads", 4, MAIN, "CPU threads a render may use. More rarely renders faster: on an 8-core CPU, 4 threads came within 2% of all 16 and used 40% less CPU, which keeps the PC responsive. 0 uses them all."),
    field("diffusion_model", MAIN_MODEL, MAIN,
          "The model that paints the image. Setup downloads the main one. The others download from Models in All settings.",
          options=DIFFUSION),
    field("use_secondary_model", False, MAIN, "Guides with a small second model. Faster and lighter, at lower quality."),
    field("init_scale", 1000, MAIN, "Pulls the render toward the start image at every step. 0 turns this off, and {skip_steps} then decides how much of it stays."),
    field("ViTB32", True, MAIN, "Another model that compares the image with your prompt. Adds VRAM use and time."),
    field("ViTB16", True, MAIN, "Another model that compares the image with your prompt. Adds VRAM use and time."),
    field("ViTL14", False, MAIN, "Compares the image with your prompt. Adds detail and uses much more VRAM."),
    field("RN50", True, MAIN, "Another model that compares the image with your prompt. Adds VRAM use and time."),
    field("image_prompts", "", MAIN, "A folder of images that guides the render beside the text, as one more prompt. Drop images or a folder on Image prompts under Guide images to make one. Every {clip_lower} reads each image, and what the images have in common counts: a subject, a style, a palette. Like an embedding, but nothing is trained. Up to 200 images. Empty is off."),
    field("image_prompt_weight", 1.0, MAIN, "How much the {image_prompts} count, beside a text prompt weighted 1. Below 0 steers away from them."),

    field("tv_scale", 20000, GUIDANCE, "Smooths the image. Higher gives fewer speckles."),
    field("range_scale", 0, GUIDANCE, "Pulls colors that run past black or white back into range, harder the further they go. In the original Disco Diffusion it did nothing, because of a bug Nano fixes, so 0 matches its look. {sat_scale} does a similar job more gently."),
    field("sat_scale", 22000, GUIDANCE, "Limits oversaturation."),
    field("clamp_grad", True, GUIDANCE, "Limits how far one step can change the image."),
    field("clamp_max", 0.065, GUIDANCE, "The limit used by {clamp_grad}. Higher is bolder."),
    field("fuzzy_prompt", False, GUIDANCE, "Adds noisy copies of each prompt for looser results."),
    field("rand_mag", 0.05, GUIDANCE, "How much noise {fuzzy_prompt} adds."),

    field("cutn_batches", 1, CUTS, "How many times the {cuts_lower} run each step. Each extra round adds that much time."),
    field("skip_augs", False, CUTS, "Turns off random flips, shifts and color changes on the {cuts_lower}."),
    field("cut_overview", "[12]*400+[4]*600", CUTS, "Whole-image {cuts_lower} over the 1000-step schedule."),
    field("cut_innercut", "[4]*400+[12]*600", CUTS, "Close-up {cuts_lower} over the 1000-step schedule."),
    field("cut_ic_pow", "[1]*400+[2]*200+[4]*200+[10]*200", CUTS, "Higher makes close-ups smaller. The default starts with close-ups of every size, which lays out the scene, and tightens them toward the end for detail."),
    field("cut_icgray_p", "[0.2]*400+[0]*600", CUTS, "Share of close-ups made grayscale."),
    field("views_fit", "Start images", CUTS, "When a picture starts partway, squeezes the {cuts_lower} schedules into the steps that are painted, so its {views_shapes} counts only those steps and it still lays out shapes before its detail. Start images fits a still's start image. Start images and clip frames also fits each frame of a clip, which repaints the video or the last frame more freely. Off keeps the schedules as drawn, so a picture that starts past 40% of the schedule gets only the detail part.", options=["Start images", "Start images and clip frames", "Off"]),
    field("views_shapes", 40, CUTS, "The share of the steps, in percent, that the first 40% of the {cuts_lower} schedules is spread over: the part that lays out the scene. When {views_fit} fits the picture, it's a share of the painted steps. 40 keeps the schedules as drawn. Lower gives more steps to detail, and higher more to big shapes."),

    field("diffusion_sampling_mode", "ddim", SAMPLING, "ddim is the standard. plms can work with fewer steps.", options=["ddim", "plms"]),
    field("eta", 0.8, SAMPLING, "Noise added each step with ddim. 0 is smoother and 1 is grainier."),
    field("randomize_class", True, SAMPLING, "Has no effect here. None of the models in Nano use ImageNet classes."),
    field("clip_denoised", False, SAMPLING, "Clamps each predicted image to the valid range."),
    field("display_rate", 1, SAMPLING, "Steps between preview updates. 1 shows every step."),
    field("keep_models_loaded", True, SAMPLING, "Keeps the models of the last render waiting in RAM, so the next render skips loading them, which saves about 20 seconds and up to 2 minutes with an SDXL handoff model. They stay until the app closes, or a render needs a different setup. Off frees the RAM after every render."),
    field("save_progress_video", False, SAMPLING, "Saves an MP4 of the picture forming beside each image, one frame per preview. Not for animation."),

    field("RN101", False, CLIP_MODELS, "Another model that compares the image with your prompt. Adds VRAM use and time."),
    field("RN50x4", False, CLIP_MODELS, "Compares the image with your prompt. Uses a lot of VRAM."),
    field("RN50x16", False, CLIP_MODELS, "Compares the image with your prompt. Uses a lot of VRAM."),
    field("RN50x64", False, CLIP_MODELS, "Compares the image with your prompt. Too large for most 8GB cards."),

    field("use_horizontal_symmetry", False, SYMMETRY, "Mirrors the left half onto the right. ddim only."),
    field("use_vertical_symmetry", False, SYMMETRY, "Mirrors the top half onto the bottom. ddim only."),
    field("transformation_percent", [0.09], SYMMETRY, "When mirroring happens, as fractions of the run.", kind="raw"),

    field("tile_scale", "Off", TILES, "Builds a larger image from several renders at the frame size. Time grows with the number of tiles.",
          options=["Off", "2x, 4 tiles", "3x, 9 tiles", "4x, 16 tiles"]),
    field("tile_denoise", 0.5, TILES, "How much each tile may change. Low keeps the picture, high paints something new."),
    field("tile_start", "Render first", TILES,
          "With no start image. Render first paints one image at the frame size, then tiles repaint it using Tile denoise, which keeps one scene with large shapes. Split the noise paints at full size from the first step, which is busier and can repeat the subject in each tile.",
          options=["Render first", "Split the noise"]),
    field("tile_whole_view", True, TILES, "CLIP also sees each tile inside a small copy of the whole picture, which helps with one scene instead of one per tile."),
    field("tile_overlap", 96, TILES, "Pixels shared by neighboring tiles. More hides joins better and gives a slightly smaller image."),

    field("handoff_file", "", HANDOFF, "The full path of a Stable Diffusion 1.5 or SDXL checkpoint, a .safetensors file. It paints the start of every image or frame, and Disco finishes it. Empty is off."),
    field("handoff_share", 30, HANDOFF, "Checkpoint first: how much of each image the {handoff_file} paints before Disco carries on. 0 turns this part off. About 30 keeps Disco's look on a sturdier layout."),
    field("handoff_denoise", 0, HANDOFF, "Denoise for Disco after the {handoff_file}: how much of its picture Disco repaints. 0 carries on from where the {handoff_file} stopped. Higher gives Disco more of the painting."),
    field("handoff_finish", 0, HANDOFF, "Reverse handoff: after Disco is done, the {handoff_file} repaints this much of the finished image, like Denoise. 0 is off. 20 to 35 tidies details and keeps Disco's look. 100 replaces Disco's image. Still images only."),
    field("handoff_steps", 30, HANDOFF, "Steps for the {handoff_file}'s whole path. It takes only its share of them. Few-step models such as Lightning, Hyper or Turbo want 4 to 8."),
    field("handoff_guidance", 7.0, HANDOFF, "How closely the {handoff_file} follows the prompt. 5 to 8 suits most checkpoints, and 1 to 2 suits few-step models."),
    field("handoff_few_steps", False, HANDOFF, "Spaces the {handoff_file}'s steps the way Lightning, Hyper, Turbo and other few-step models expect."),
    field("handoff_negative", "blurry, lowres, watermark, text", HANDOFF, "What the {handoff_file} steers away from. Prompt lines with a negative weight are added to it."),
    field("small_first", 0, HANDOFF, "Small first: Disco paints this share of each image smaller, in the frame's shape, and hands its still blurry picture to itself at full size. Disco's model learned at about 512 pixels, so a large frame can grow the subject more than once. The small start lays out one scene, and the full size adds the detail. 0 is off. About 25 suits 1280x768. No checkpoint needed."),
    field("small_denoise", 0, HANDOFF, "Denoise for the full size after {small_first}: how much of the small picture it repaints. 0 carries on from where the small start stopped, and so does anything below 100 minus {small_first}. Higher gives the full size more of the layout: 85 with {small_first} at 25 starts it at step 31 of 200."),
    field("small_hold", 0.0, HANDOFF, "After {small_first}, holds the full-size render's broad shapes and colors to the small picture while the full size adds detail, so a second subject can't grow in the empty areas. Fine detail is left free. About 0.5 is a good start. 0 is off."),
    field("small_hold_curve", "[1]*500+[0]*500", HANDOFF, "How much {small_hold} applies, over the 1000-step schedule. The default holds until halfway, then lets go for the detail."),
    field("small_side", 768, HANDOFF,"The longer side of the {small_first} start, in pixels. The shorter side follows the frame's shape. It does nothing on frames no larger than this."),
    field("handoff_keep", False, HANDOFF,"Also saves the {handoff_file}'s picture at the handoff beside each image. Not for animation."),

    field("animation_mode", "None", ANIMATION, "2D moves a flat picture. 3D flies a camera through the scene, using Depth Anything to tell near from far. Video Input repaints every frame of a video.",
          options=["None", "2D", "3D", "Video Input"]),
    field("max_frames", 60, ANIMATION, "How many frames to render. Video Input stops at the end of the video."),
    field("fps", 12, ANIMATION, "Frames per second in the saved video."),
    field("frames_skip_steps", 75, ANIMATION, "The share of {steps} skipped on each frame after the first, in percent. Higher keeps more of the last frame and renders faster. 75 holds a scene while it moves. 60 lets it keep reinventing itself. Video Input often wants 45 to 60, so the video changes more."),
    field("frame_skip_keys", "", ANIMATION, "Changes {frames_skip_steps} over time, with keyframes such as 0:(75), 20:(55), 32:(75). Dropping it around a prompt change lets the new prompt take over faster. Empty uses {frames_skip_steps} throughout."),
    field("frames_scale", 0, ANIMATION, "Pulls each frame toward the moved last frame, like {init_scale}. Steadier, with a little more VRAM. 0 is off."),
    field("color_hold", 0.5, ANIMATION, "Keeps the colors close to the first frame's, which stops them drifting over a long clip. 0 is off and 1 holds them fully."),
    field("frame_sharpen", 0.0, ANIMATION, "2D and 3D. Sharpens each moved frame, since every move softens it a little. It builds up over a clip into ringing edges, especially at a high {frames_skip_steps}, so keep it small. 0 is off."),
    field("frame_ghost", False, ANIMATION, "Each frame uses the moved last frame as its ghost, so {ghost_lowfreq} and {depth_target} hold the scene steady."),
    field("prompt_fade", 0, ANIMATION, "Frames over which a new prompt set fades in. 0 switches at once. Start a line with 120: to begin a new set at frame 120."),
    field("zoom", "0:(1.02)", ANIMATION, "2D. Above 1 moves in each frame, below 1 moves out. " + "Keyframes such as 0:(1.0), 60:(1.02) change it over time, and formulas of the frame t such as 0:(1+0.02*sin(t/8)) work too."),
    field("angle", "0:(0)", ANIMATION, "2D. Degrees the picture turns each frame, counterclockwise. " + "Keyframes such as 0:(1.0), 60:(1.02) change it over time, and formulas of the frame t such as 0:(1+0.02*sin(t/8)) work too."),
    field("translation_x", "0:(0)", ANIMATION, "2D and 3D. Moves the camera right each frame, or left below 0. In pixels for 2D. " + "Keyframes such as 0:(1.0), 60:(1.02) change it over time, and formulas of the frame t such as 0:(1+0.02*sin(t/8)) work too."),
    field("translation_y", "0:(0)", ANIMATION, "2D and 3D. Moves the camera down each frame, or up below 0. In pixels for 2D. " + "Keyframes such as 0:(1.0), 60:(1.02) change it over time, and formulas of the frame t such as 0:(1+0.02*sin(t/8)) work too."),
    field("translation_z", "0:(10.0)", ANIMATION, "3D. Flies the camera forward each frame, or back below 0. " + "Keyframes such as 0:(1.0), 60:(1.02) change it over time, and formulas of the frame t such as 0:(1+0.02*sin(t/8)) work too."),
    field("rotation_3d_x", "0:(0)", ANIMATION, "3D. Degrees the camera tilts each frame. Above 0 looks up. " + "Keyframes such as 0:(1.0), 60:(1.02) change it over time, and formulas of the frame t such as 0:(1+0.02*sin(t/8)) work too."),
    field("rotation_3d_y", "0:(0)", ANIMATION, "3D. Degrees the camera turns each frame. Above 0 turns right. " + "Keyframes such as 0:(1.0), 60:(1.02) change it over time, and formulas of the frame t such as 0:(1+0.02*sin(t/8)) work too."),
    field("rotation_3d_z", "0:(0)", ANIMATION, "3D. Degrees the camera rolls each frame. " + "Keyframes such as 0:(1.0), 60:(1.02) change it over time, and formulas of the frame t such as 0:(1+0.02*sin(t/8)) work too."),
    field("interp_spline", "Linear", ANIMATION, "How values move between keyframes. Linear runs straight, and Cubic curves smoothly through each key.",
          options=["Linear", "Cubic"]),
    field("fov", 40, ANIMATION, "3D. The camera's field of view in degrees. Wider makes depth stronger."),
    field("padding_mode", "border", ANIMATION, "What fills the edges a move uncovers. border stretches the edge pixels, reflection mirrors the picture.",
          options=["border", "reflection"]),
    field("sampling_mode", "bilinear", ANIMATION, "How a moved frame is resampled. bicubic is sharper for a frame, but its slight overshoot builds up over a clip into halos along every edge. Disco Diffusion used bicubic.", options=["bilinear", "bicubic"]),
    field("video_init_path", "", ANIMATION, "The full path of the video for Video Input."),
    field("extract_nth_frame", 2, ANIMATION, "Video Input uses every nth frame of the video. 1 uses all of them."),
    field("video_init_seed_continuity", True, ANIMATION, "Video Input. Every frame starts from the same noise, which flickers less."),
    field("in_between", 0, ANIMATION, "Adds this many frames between each pair of painted frames, so motion is smoother. The clip keeps its length, because {fps} is multiplied to match. RIFE paints them if you get it from Models in All settings. Otherwise optical flow does, which can ghost on fast motion. 0 is off."),
    field("loop_mode", "Off", ANIMATION, "Makes the clip loop with no jump where it starts again. Paint back to the start paints the last {loop_frames} frames more and more like the first, for motion that already comes back to where it began, such as a camera moving on a sin formula. Ping-pong plays the video, or the camera's moves, forward and then back, so the motion ends where it began, and the way back is painted fresh. With {in_between}, the last frame gets in-betweens into the first too.",
          options=["Off", "Paint back to the start", "Ping-pong"]),
    field("loop_frames", 6, ANIMATION, "With {loop_mode}, how many of the last frames are painted back toward the start, more each frame. More is smoother and spends more of the clip coming back."),
    field("video_blend", 0.4, ANIMATION, "Video Input. Mixes the last finished frame into each new one, so the painting builds up over the clip instead of starting again from the video every frame. With {video_flow} on, the paint moves with the video. 0 is off."),
    field("video_flow", True, ANIMATION, "Video Input. Moves the last finished frame along with the video's own motion before {video_blend} mixes it in, so paint travels with what moves instead of trailing."),
    field("turbo_mode", False, ANIMATION, "2D and 3D. Only every few frames is repainted, and the camera moves the frames between. Much faster."),
    field("turbo_steps", 3, ANIMATION, "With {turbo_mode}, how often a frame is repainted."),
    field("save_mp4", True, ANIMATION, "Saves an MP4 video."),
    field("save_webp", True, ANIMATION, "Saves an animated WebP, which browsers play like a GIF at much better quality."),
    field("save_gif", False, ANIMATION, "Saves a GIF at most 512 pixels wide. GIFs are large and have only 256 colors."),

    field("time_travel", False, LAB, "Backs up and repeats some steps, so the painter can absorb the prompt's push. Cleaner subjects, about 35% slower."),
    field("prompt_graft", 0.0, LAB, "Adds the effect of the prompt straight into Disco's own decoder, translated from the partner model. About 3 is a good start. 0 is off."),
    field("borrow_attention", False, LAB, "Disco borrows the partner model's sense of which parts belong together. Better subjects, often grayer."),
    field("partner_model", "Tiny SD", LAB,
          "The small text model behind {prompt_graft}, {live_ghost}, {two_painters} and {borrow_attention}. It never paints the final image. Tiny SD is used unless you choose SD 1.5 file here to use a checkpoint of your own, set in {partner_file}. {prompt_graft} always uses Tiny SD. Get Tiny SD from Models in All settings.",
          options=["Tiny SD", "SD 1.5 file"]),
    field("detail_bands", False, LAB, "The prompt's push keeps to broad shapes early and adds fine detail later."),
    field("shaped_guidance", False, LAB, "Changes {clip_guidance_scale} over the render, using its curve."),
    field("gradient_momentum", 0.0, LAB, "Averages the prompt's push across steps. Higher turns wilder and more abstract. 0 is off."),
    field("evidence_map", 0.0, LAB, "Remembers where the subject is forming and eases the push elsewhere. 0 is off."),
    field("time_travel_curve", "[0]*300+[1]*300+[0]*400", LAB, "Extra tries at each step, over the 1000-step schedule. Each one adds a step of time."),
    field("guidance_curve", "[0.5]*250+[1.5]*400+[1]*250+[0.3]*100", LAB, "Multiplies {clip_guidance_scale}, over the 1000-step schedule."),
    field("detail_curve", "[0.2]*500+[1]*500", LAB, "How much fine detail the prompt's push keeps, over the 1000-step schedule. 0 keeps only broad shapes."),
    field("live_ghost", "Off", LAB, "The partner model supplies the ghost. Sketch first paints it before the render. Live keeps repainting it during the render.",
          options=["Off", "Sketch first", "Live"]),
    field("control_image", "", CONTROLS, "For a still, the picture whose outlines, depth, pose and people the render follows, or drop one on Control image under Controls. Empty uses the start image. A clip doesn't use it: it follows each frame of its video, or the frame each repaint starts from."),
    field("edge_control", 0.0, CONTROLS, "Holds the control picture's outlines. It only asks for an outline where the control has one, so plain areas stay free for the prompt to paint. About 0.5 is a good start. 0 is off."),
    field("depth_control", 0.0, CONTROLS, "Holds the control picture's near and far, so the same things stand in front. About 0.5 is a good start. Needs Depth Anything from Models in All settings. 0 is off."),
    field("pose_control", 0.0, CONTROLS, "Finds the people in the control picture and holds a body in each one's pose, whatever it is painted as. About 0.5 is a good start. Downloads Keypoint R-CNN, 0.24GB, the first time. 0 is off."),
    field("person_control", 0.0, CONTROLS, "Cuts out the people in the control picture and holds their silhouettes, clothes and broad colors, while everything around them is painted freely. Fine detail inside them stays free too, so the prompt can still paint over them. About 0.5 is a good start. Downloads Mask R-CNN, 0.19GB, the first time. 0 is off."),
    field("face_control", 0.0, CONTROLS, "Finds the faces in the control picture and holds each one's features: where the eyes, nose and mouth are, and their light and dark, so the face and its expression stay while its skin, color and brushwork are painted freely. About 0.5 is a good start. Downloads RetinaFace, 0.11GB, the first time. 0 is off."),
    field("face_shape", 0.0, CONTROLS, "Holds each face's outline and its near and far from the control picture, measured close up, so the face keeps its own shape and a turned head keeps more of its turn, most of all with a Control image. Where the eyes look is too fine to hold. It pushes beside the prompt, as Edge and Depth control do, so the features are still painted freely, though the skin comes out a little smoother. Use it with {face_control} or on its own. About 0.5 is a good start; at 1 it can leave bright patches. Needs Depth Anything from Models in All settings, and downloads RetinaFace, 0.11GB, the first time. 0 is off."),
    field("face_detail", 0, CONTROLS, "Close-ups each face gets at every step, beside the usual ones, so faces are painted with more care. Faces come from the control picture, or with none, from the render itself once one starts to show, so this works with text alone. About 2 is a good start, at about 5% more time. 0 is off."),
    field("edge_curve", "[1]*800+[0]*200", CONTROLS, "How much {edge_control} counts, over the 1000-step schedule. Outlines still matter while the detail forms, so it holds late."),
    field("depth_curve", "[1]*600+[0]*400", CONTROLS, "How much {depth_control} counts, over the 1000-step schedule. Near and far are settled early, so it lets go sooner."),
    field("pose_curve", "[1]*700+[0]*300", CONTROLS, "How much {pose_control} counts, over the 1000-step schedule."),
    field("face_curve", "[1]*500+[0]*500", CONTROLS, "How much {face_control} counts, over the 1000-step schedule. It lets go halfway: held later, it fights the features as they are painted and leaves burned patches."),
    field("face_shape_curve", "[1]*700+[0]*300", CONTROLS, "How much {face_shape} counts, over the 1000-step schedule. Which way a head is turned is settled with the shapes, so it lets go before the finest detail."),
    field("person_curve", "[1]*700+[0]*300", CONTROLS, "How much {person_control} counts, over the 1000-step schedule. Held late, the person looks pasted in rather than painted."),
    field("depth_target", 0.0, LAB, "Steers the render toward the depth of the ghost, so near and far stay in place. About 1 is a good start. It needs {live_ghost} or {ghost_image}. Get Depth Anything from Models in All settings. 0 is off."),
    field("two_painters", False, LAB, "The partner model and Disco paint one picture at every step. The partner owns the broad shapes, and the size it owns shrinks over the render."),
    field("painters_curve", "[64]*300+[32]*300+[16]*200+[0]*200", LAB, "The smallest shape, in pixels, that the partner owns, over the 1000-step schedule. 0 leaves everything to Disco."),
    field("painters_strength", 1.0, LAB, "How much of the broad shapes the partner decides."),
    field("painters_color", False, LAB, "Also takes the partner's broad color, not just its light and dark."),
    field("painters_path", "Blend", LAB, "Blend mixes the two predictions. Replace moves the picture straight to the mix.", options=["Blend", "Replace"]),
    field("graft_curve", "[1]*700+[0]*300", LAB, "How much of the graft is used, over the 1000-step schedule."),
    field("sketch_steps", 25, LAB, "Steps the partner model takes for a sketch painted first."),
    field("partner_file", "", LAB, "The full path of an SD 1.5 .safetensors file, for {partner_model}."),
    field("borrow_curve", "[0.5]*700+[0]*300", LAB, "How much of the borrowed attention is used, over the 1000-step schedule."),
    field("partner_every", 4, LAB, "Steps between the partner model's updates."),
    field("partner_follow", 0.5, LAB, "How closely the partner model follows the painting. 0 lets it paint its own picture."),
    field("partner_keep_sketch", False, LAB, "Also saves the partner model's own rough picture beside the render."),
    field("ghost_image", "", LAB, "The full path of an image whose layout guides the render, or drop one on Ghost image under Guide images. It is never mixed in as pixels."),
    field("ghost_structure", 0.0, LAB, "How much the render copies the ghost's shapes, not its look."),
    field("ghost_crops", 0.0, LAB, "How much each area should match the same area of the ghost, as CLIP sees it."),
    field("ghost_lowfreq", 0.3, LAB, "How much of the ghost's broad light and dark is moved into the render."),
    field("ghost_curve", "[1]*700+[0]*300", LAB, "How much the ghost counts, over the 1000-step schedule."),
    field("style_image", "", LAB, "The full path of an image whose texture and palette guide the render, or drop one on Style image under Guide images."),
    field("style_strength", 1.0, LAB, "How strongly the style image counts."),
    field("route_prompts", "Off", LAB, "Scores the subject of your prompt on whole-image views and its style on close-ups, so the subject forms once and the paint covers everything. Subject everywhere also scores the subject on close-ups. {route_split} says which part is the subject.",
          options=["Off", "Split", "Subject everywhere"]),
    field("route_split", "First phrase", LAB, "Which part of the prompt is the subject. First phrase: everything before the first comma of each line. First line: the whole first line, with the other lines as style.",
          options=["First phrase", "First line"]),
    field("route_curve", "[1]*500+[0]*500", LAB, "How much {route_prompts} applies, over the 1000-step schedule. 0 scores every prompt on every view, as usual."),
    field("layout_critic", 0.0, LAB, "Scores each view again with its pieces shuffled, and pushes the picture so the shuffled view scores worse. The prompt then only counts when it comes from how the parts are arranged, which holds subjects together. About 0.6 is a good start. Adds about 10% time and 0.8 to 1.2GB of VRAM. 0 is off."),
    field("layout_cuts", "All", LAB, "Which views {layout_critic} shuffles. Whole-image costs less VRAM.", options=["All", "Whole-image"]),
    field("layout_margin", 0.02, LAB, "How much worse a shuffled view has to score before {layout_critic} stops pushing. Higher keeps pushing for longer."),
    field("layout_curve", "[1]*400+[0]*600", LAB, "How much {layout_critic} applies, over the 1000-step schedule."),
    field("robust_critic", "Off", LAB, "A checker trained to resist tiny changes, so its push draws real shapes instead of streaks. It leads early, while the layout forms, and hands over to the usual checkers. Whole scenes and steadier subjects, with more muted color. Get one from Models in All settings.",
          options=["Off", "FARE4 B/16", "FARE2 L/14", "FARE4 L/14"]),
    field("robust_mode", "Hand over", LAB, "Hand over fades the usual checkers in as the robust one fades out. Together keeps the usual checkers at full strength all along.",
          options=["Hand over", "Together"]),
    field("robust_curve", "[1]*300+[0.75]*50+[0.5]*50+[0.25]*50+[0.1]*50+[0]*500", LAB, "How much the robust checker counts, over the 1000-step schedule."),
    field("one_subject", 0.0, LAB, "Finds where each subject is in the picture and pushes it into one place, so it isn't repeated, and keeps separate subjects apart. About 1 for one subject, 2 for two. Needs a ViT checker. 0 is off."),
    field("subject_words", "", LAB, "The subjects for {one_subject}, separated by semicolons, such as: a woman; a white horse. Empty uses the first phrase of the prompt."),
    field("one_subject_curve", "[1]*500+[0]*500", LAB, "How much {one_subject} applies, over the 1000-step schedule."),
    field("correct_steps", 0, LAB, "Extra small steps after each step early in the render, which settle the picture before it moves on. With {two_painters}, they also make both models agree. Each one adds about 30% time. 0 is off."),
    field("correct_curve", "[1]*400+[0]*600", LAB, "Where {correct_steps} run, over the 1000-step schedule."),
    field("correct_snr", 0.1, LAB, "How far each of {correct_steps} moves. Higher changes more and can turn noisy."),
    field("correct_clip", True, LAB, "{correct_steps} also follow the prompt. Off, they only settle the painting."),
    field("borrow_grids", "Coarse", LAB, "Which of Disco's attention layers borrow. Coarse keeps Disco's color. All gives a stronger, grayer result. Coarsest changes the least.",
          options=["Coarse", "All", "Coarsest"]),
]

# Plain names. The official names are the field names, which match Disco Diffusion's.
LABELS = {
    "steps": "Steps", "skip_steps": "Denoise", "clip_guidance_scale": "Prompt strength", "n_batches": "Images", "set_seed": "Seed",
    "tile_scale": "Tiles", "tile_denoise": "Tile denoise", "tile_start": "Text start", "tile_whole_view": "Whole picture view",
    "tile_overlap": "Tile overlap", "exact_repeat": "Exact repeat", "cpu_threads": "CPU threads", "diffusion_model": "Model",
    "use_secondary_model": "Secondary model", "init_scale": "Start image strength",
    "image_prompts": "Image prompts", "image_prompt_weight": "Image prompt weight",
    "small_first": "Small first %", "small_denoise": "Full-size denoise %", "small_side": "Small first size",
    "small_hold": "Hold the small layout", "small_hold_curve": "Hold curve",
    "ViTB32": "Broad checker", "ViTB16": "Fine checker", "ViTL14": "Finest checker", "RN50": "ResNet checker",
    "RN101": "Deeper ResNet checker", "RN50x4": "Large ResNet checker", "RN50x16": "Very large ResNet checker",
    "RN50x64": "Huge ResNet checker",
    "tv_scale": "Smoothing", "range_scale": "Color range", "sat_scale": "Saturation limit", "clamp_grad": "Limit each step",
    "clamp_max": "Step limit", "fuzzy_prompt": "Loose prompt", "rand_mag": "Looseness",
    "cutn_batches": "View rounds", "skip_augs": "Plain views", "cut_overview": "Whole-image views",
    "cut_innercut": "Close-up views", "cut_ic_pow": "Close-up tightness", "cut_icgray_p": "Gray close-ups",
    "views_fit": "Fit views to the painted steps", "views_shapes": "Shapes share %",
    "diffusion_sampling_mode": "Sampler", "eta": "Step noise", "randomize_class": "Random class",
    "clip_denoised": "Keep colors in range", "display_rate": "Preview every", "save_progress_video": "Save progress video", "keep_models_loaded": "Keep models loaded",
    "use_horizontal_symmetry": "Mirror left and right", "use_vertical_symmetry": "Mirror top and bottom",
    "transformation_percent": "Mirror timing",
    "time_travel": "Time travel", "partner_model": "Partner model", "detail_bands": "Broad shapes first",
    "shaped_guidance": "Shaped prompt strength", "gradient_momentum": "Push momentum", "evidence_map": "Subject memory",
    "time_travel_curve": "Time travel curve", "guidance_curve": "Prompt strength curve", "detail_curve": "Broad shapes curve",
    "control_image": "Control image", "edge_control": "Edge control", "depth_control": "Depth control", "pose_control": "Pose control", "person_control": "Person control",
    "face_control": "Face control", "face_detail": "Face detail", "face_curve": "Face curve", "face_shape": "Face shape", "face_shape_curve": "Face shape curve",
    "edge_curve": "Edge curve", "depth_curve": "Depth curve", "pose_curve": "Pose curve", "person_curve": "Person curve",
    "live_ghost": "Live ghost", "depth_target": "Depth target", "two_painters": "Two painters", "painters_curve": "Two painters curve",
    "painters_strength": "Two painters strength", "painters_color": "Two painters color", "painters_path": "Two painters path",
    "prompt_graft": "Prompt graft", "graft_curve": "Prompt graft curve", "sketch_steps": "Sketch steps", "borrow_attention": "Borrowed attention",
    "partner_file": "SD 1.5 file", "borrow_curve": "Borrowed attention curve", "partner_every": "Partner every", "partner_keep_sketch": "Keep the partner's sketch",
    "partner_follow": "Follow the painting", "ghost_image": "Ghost image", "ghost_structure": "Ghost shapes",
    "ghost_crops": "Ghost areas", "ghost_lowfreq": "Ghost light and dark", "ghost_curve": "Ghost curve",
    "style_image": "Style image", "style_strength": "Style strength",
    "route_prompts": "Prompt routing", "route_split": "Subject is", "route_curve": "Prompt routing curve",
    "layout_critic": "Coherence critic", "layout_cuts": "Coherence views", "layout_margin": "Coherence margin", "layout_curve": "Coherence curve",
    "robust_critic": "Robust checker", "robust_mode": "Robust checker mode", "robust_curve": "Robust checker curve",
    "one_subject": "One subject", "subject_words": "Subjects", "one_subject_curve": "One subject curve",
    "correct_steps": "Corrector steps", "correct_curve": "Corrector curve", "correct_snr": "Corrector size", "correct_clip": "Corrector follows the prompt",
    "borrow_grids": "Borrowed layers",
    "handoff_file": "Handoff model", "handoff_share": "Handoff %", "handoff_denoise": "Disco denoise %", "handoff_finish": "Reverse handoff %", "handoff_steps": "Handoff steps",
    "handoff_guidance": "Handoff prompt strength", "handoff_few_steps": "Few-step model", "handoff_negative": "Handoff negative prompt",
    "handoff_keep": "Keep the handoff picture",
    "animation_mode": "Animation", "max_frames": "Frames", "fps": "Frames per second", "frames_skip_steps": "Frame skip %", "frame_skip_keys": "Frame skip keyframes",
    "frames_scale": "Frame strength", "color_hold": "Hold colors", "frame_sharpen": "Sharpen moves", "frame_ghost": "Last frame as ghost", "prompt_fade": "Prompt fade",
    "zoom": "Zoom", "angle": "Turn", "translation_x": "Move right", "translation_y": "Move down", "translation_z": "Move forward",
    "rotation_3d_x": "Tilt", "rotation_3d_y": "Pan", "rotation_3d_z": "Roll", "interp_spline": "Keyframe shape", "fov": "Field of view",
    "padding_mode": "Edge fill", "sampling_mode": "Resampling", "video_init_path": "Video file", "extract_nth_frame": "Every nth frame",
    "video_init_seed_continuity": "Same noise each frame", "video_blend": "Blend last frame", "loop_mode": "Seamless loop", "loop_frames": "Loop back over", "in_between": "In-between frames", "video_flow": "Follow the video's motion", "turbo_mode": "Turbo",
    "turbo_steps": "Turbo every", "save_mp4": "Save MP4", "save_webp": "Save WebP", "save_gif": "Save GIF",
}

# Nano's own settings, which have no official name.
NANO_ONLY = {"views_fit", "views_shapes", "exact_repeat", "cpu_threads", "save_progress_video", "keep_models_loaded", "image_prompts", "image_prompt_weight", "tile_scale", "tile_denoise", "tile_start", "tile_whole_view", "tile_overlap",
             "color_hold", "frame_sharpen", "frame_skip_keys", "frame_ghost", "prompt_fade", "in_between", "loop_mode", "loop_frames", "video_blend", "video_flow", "save_mp4", "save_webp", "save_gif",
             *(f["name"] for f in FIELDS if f["section"] in (LAB, HANDOFF, CONTROLS))}

# Words that aren't settings, as plain and official.
WORDS = {
    "prompt": ("Prompt", "text_prompts"),
    "batch_name": ("Folder name", "batch_name"),
    "init_image": ("Start from an image", "init_image"),
    "clip_models": ("Prompt checkers", "CLIP models"),
    "clip_lower": ("prompt checkers", "CLIP models"),
    "cuts_lower": ("views", "cutouts"),
    CUTS: ("Views", CUTS),
    CLIP_MODELS: ("More prompt checkers", CLIP_MODELS),
    # Denoise is shown as a share of the steps, skip_steps as a step count.
    "skip_low": ("1%", "0"),
    "skip_default": ("100%", "0"),
    "skip_high": ("100%", "steps - 1"),
}

# The ends of each setting's useful range, shown in Help. Schedules give the value for one step.
RANGES = {
    "steps": (25, 1000),
    "skip_steps": ("{skip_low}", "{skip_high}"),
    "clip_guidance_scale": (1500, 100000),
    "n_batches": (1, 100),
    "set_seed": (0, 4294967295),
    "cpu_threads": (0, 64),
    "init_scale": (0, 10000),
    "image_prompt_weight": (-2, 4),
    "small_first": (0, 60),
    "small_denoise": (0, 100),
    "small_hold": (0, 1),
    "small_side": (384, 1024),
    "tv_scale": (0, 100000),
    "range_scale": (0, 100000),
    "sat_scale": (0, 100000),
    "clamp_max": (0.05, 0.3),
    "rand_mag": (0, 1),
    "cutn_batches": (1, 8),
    "cut_overview": (0, 32),
    "cut_innercut": (0, 32),
    "cut_ic_pow": (0.5, 100),
    "cut_icgray_p": (0, 1),
    "views_shapes": (0, 100),
    "eta": (0, 1),
    "display_rate": (1, "{steps}"),
    "transformation_percent": (0, 1),
    "tile_denoise": (0.1, 1),
    "tile_overlap": (32, 128),
    "gradient_momentum": (0, 0.95),
    "evidence_map": (0, 1),
    "partner_every": (1, 25),
    "painters_strength": (0, 1),
    "sketch_steps": (5, 100),
    "prompt_graft": (0, 20),
    "partner_follow": (0, 1),
    "ghost_structure": (0, 3),
    "ghost_crops": (0, 3),
    "ghost_lowfreq": (0, 1),
    "depth_target": (0, 3),
    "edge_control": (0, 2),
    "face_shape": (0, 2),
    "depth_control": (0, 2),
    "pose_control": (0, 2),
    "person_control": (0, 1),
    "face_control": (0, 1),
    "face_detail": (0, 8),
    "style_strength": (0, 3),
    "layout_critic": (0, 2),
    "layout_margin": (0, 0.3),
    "one_subject": (0, 3),
    "correct_steps": (0, 3),
    "correct_snr": (0.02, 0.3),
    "handoff_share": (0, 100),
    "handoff_denoise": (0, 100),
    "handoff_finish": (0, 100),
    "handoff_steps": (4, 60),
    "handoff_guidance": (1, 12),
    "max_frames": (1, 10000),
    "fps": (1, 60),
    "frames_skip_steps": (30, 90),
    "frames_scale": (0, 10000),
    "color_hold": (0, 1),
    "frame_sharpen": (0, 0.5),
    "prompt_fade": (0, 120),
    "fov": (20, 120),
    "extract_nth_frame": (1, 10),
    "video_blend": (0, 0.7),
    "in_between": (0, 3),
    "loop_frames": (1, 30),
    "turbo_steps": (2, 6),
}

for f in FIELDS:
    f["label"] = LABELS.get(f["name"], f["name"])
    f["low"], f["high"] = RANGES.get(f["name"], (None, None))
    if f["name"] == "skip_steps":
        f["shown_default"] = "{skip_default}"

DEFAULTS = {f["name"]: f["default"] for f in FIELDS}
TERMS = {**WORDS, **{f["name"]: (f["label"], f["label"] if f["name"] in NANO_ONLY else f["name"]) for f in FIELDS}}


def uses_partner(settings):
    return bool(settings.get("borrow_attention") or settings.get("two_painters") or settings.get("prompt_graft")
                or settings.get("live_ghost", "Off") != "Off")


def small_frame(width, height, longest):
    """The size Small first paints at: the frame's shape in multiples of 64, its longer side at most `longest`.
    Of the sizes with at least 60% of the area that allows, the largest whose shape is within 2% of the frame's,
    or else the closest shape, so the picture is barely stretched when it is enlarged. 1280x768 gives 640x384.
    None when the frame is no larger than `longest`."""
    long, short = max(width, height), min(width, height)
    if long <= longest:
        return None
    ratio, found = long / short, []
    top = 64 * max(longest // 64, 1)
    for side in range(top, 63, -64):
        other = max(round(side / ratio / 64), 1) * 64
        if side * other >= 0.6 * top * top / ratio or not found:
            found.append((abs(side / other / ratio - 1), -side, other))
    close = [f for f in found if f[0] <= 0.02]
    _, side, other = close[0] if close else min(found)
    side = -side
    return (side, other) if width >= height else (other, side)


def small_first_on(settings):
    """Small first is Disco's handoff to itself, so it is set aside while a checkpoint's handoff is chosen."""
    return float(settings.get("small_first", DEFAULTS["small_first"]) or 0) > 0 and not str(settings.get("handoff_file") or "")


def handoff_parts(settings):
    """Whether the handoff model paints the start and whether it finishes, as (start, finish)."""
    if not str(settings.get("handoff_file") or ""):
        return False, False
    number = lambda name: float(settings.get(name, DEFAULTS[name]) or 0)  # noqa: E731
    return number("handoff_share") > 0, number("handoff_finish") > 0
