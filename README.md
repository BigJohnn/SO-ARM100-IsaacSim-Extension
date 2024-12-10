[Warning] Use at your own risk! If you use this to control your real-life SO-ARM-100.

This is a script extension.
Just  create yours as https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorial_extension_templates.html#scripting-template says.
Then replace files within this repo.

Select mode in ui_builder.py "use_follow=False/True"

If we use follow, we are ignoring collisions, which may lead to self-intersection.
If we use RMPflow, it considers collisions and obstacle avoidance, but the related parameters for SO100 are not yet perfect. This may result in large action steps for the end effector.

And maybe the 1st method is relatively more stable.