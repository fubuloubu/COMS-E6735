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
	@rm -rf $@
	@mkdir $@
	@tar xzf $< -C $@

%s.txt: %_images
	@echo "   GEN $*s list"
	@find ./$*_images -iname "*.jpg" > $@

.PHONY: fix-positives
fix-positives: positive_images
	@echo "FIXING $^"
	@../fix_size_to_largest.py $^

GET_W_AND_H=$(shell python -c "from PIL import Image; sz=Image.open('$(1)').size; print '-w ' + str(sz[0]) + ' -h ' + str(sz[1]);")
test:
	echo "$(call GET_W_AND_H,./positive_images/positive_0047.jpg)"

POSITIVES=positives.txt
NEGATIVES=negatives.txt
LIST_FILES=$(POSITIVES) $(NEGATIVES)
samples: $(LIST_FILES) fix-positives
	@echo "   GEN $@"
	@rm -rf $@*
	@mkdir $@
	@perl "../opencv-haar-classifier-training/bin\
	/createsamples.pl" \
		$(LIST_FILES) \
		samples 1500 \
		"opencv_createsamples \
		-bgcolor 0 -bgthresh 0 \
		-maxxangle 1.1 -maxyangle 1.1 \
		maxzangle 0.5 -maxidev 40 \
		$(call GET_W_AND_H,S(shell head -n 1 $(POSITIVES)))" \
		> /dev/null
	@find ./$@ -iname "*.vec" > $@.txt

samples.vec: samples.txt
	@echo "   GEN $@"
	@#Delete the empty files for now to avoid errors TODO
	@find samples -size 0 -exec echo "   DEL empty file: {}" \;
	@find samples -size 0 -delete
	@python ../mergevec/mergevec.py -v samples -o samples.vec

classifier: samples.vec
	@echo " TRAIN $${PWD##*/}"
	@opencv_traincascade -data $@\
		-vec $< -bg $(NEGATIVES)\
		-numStages 20 -minHitRate 0.999\
	   	-maxFalseAlarmRate 0.5\
		-numPos 1000 -numNeg 600\
		$(call GET_W_AND_H,S(shell head -n 1 $(POSITIVES))) \
		-mode ALL -numThreads 7\
	   	-precalcValBufSize 1024\
		-precalcIdxBufSize 1024

INTERMEDIATES=*_frames *_images *s.txt samples samples.vec
.PHONY: clean
clean:
	@echo " CLEAN $${PWD##*/}"
	@rm -rf $(INTERMEDIATES)
