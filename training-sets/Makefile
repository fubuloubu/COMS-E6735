%_frames: %.mp4
	@echo " CREATE $@"
	@mkdir $@
	@#Overwrite output files (-y)
	@#Disable audio recording (-an)
	@#Number of frames per second (-r #)
	@#Forced format image2 (-f image2)
	@ffmpeg -i $< -y -an -r 6 -f image2 $@/$*_%04d.jpg
	@echo "Please manually select images in $@"

%.tar.gz: %_frames
	@echo "   ZIP $<"
	@tar czf $@ -C $< .

%_images: %.tar.gz
	@echo " UNZIP $<"
	@mkdir $@
	@tar xzf $< -C $@


