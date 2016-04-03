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

.SECONDARY: negative_images positive_images
%_images: | %.tar.gz
	@echo " UNZIP $|"
	@rm -rf $@
	@mkdir $@
	@tar xzf $| -C $@

# Make all input positive image files 100x100 px
fix_positives: positive_images
	@echo "FIXING $<"
	@../resize_images.py -W 100 -H 100 $<
	@# Create a fake target to avoid dependancy hell
	@touch $@
	
positives.txt: | fix_positives
%s.txt: %_images
	@echo "   GEN $*s list"
	@find ./$*_images -iname "*.jpg" > $@

# Shell expansion functions for use below
GET_W_AND_H=$(shell python -c "from PIL import Image; sz=Image.open('$(1)').size; \
			print '-w ' + str(sz[0]) + ' -h ' + str(sz[1]);")
GET_NUM_SAMPLES=$(shell wc -l $(1) | awk {'print $$1'})
MULTIPLY=$(shell expr $(1) \* $(2))
FIRST_FILE=$(shell head -n 1 $(1))

POSITIVES=positives.txt
NEGATIVES=negatives.txt
LIST_FILES=$(POSITIVES) $(NEGATIVES)

samples: $(LIST_FILES)
	@echo "   GEN $@"
	@rm -rf $@*
	@mkdir $@
	@#Create 10 times the number of positives by using createsamples utility
	@perl ../opencv-haar-classifier-training/bin/createsamples.pl $(LIST_FILES) \
		samples $(call MULTIPLY,$(call GET_NUM_SAMPLES,$(POSITIVES)),10) \
		"opencv_createsamples -bgcolor 0 -bgthresh 0 \
		-maxxangle 1.1 -maxyangle 1.1 -maxzangle 0.5 -maxidev 40 \
		$(call GET_W_AND_H,$(call FIRST_FILE,$(POSITIVES)))" \
		> /dev/null
	@find ./$@ -iname "*.vec" > $@.txt

.SECONDARY: samples
samples.vec: | samples
	@echo "   GEN $@"
	@#Delete the empty files for now to avoid errors TODO
	@find samples -size 0 -exec echo "   DEL {} (empty file)" \;
	@find samples -size 0 -delete
	@python ../mergevec/mergevec.py -v samples -o samples.vec

# Get main memory size (in kB), and return an third of that amount in megabytes for buffer sizing
BUFFER_SIZE=$(shell grep MemTotal /proc/meminfo | awk '{print $$2}' | xargs -I {} expr {} / 1024 / 3)
# Get the number of processors on current computer and cut that in half
NUM_THREADS=$(shell grep -c ^processor /proc/cpuinfo | xargs -I {} expr {} / 2)

.PHONY: classifier
classifier: samples.vec
	@echo " TRAIN $${PWD##*/}"
	@rm -rf $@
	@mkdir $@
	@opencv_traincascade -data $@ \
		-vec $< -bg $(NEGATIVES) \
		-numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -mode ALL \
		-numPos $(call MULTIPLY,$(call GET_NUM_SAMPLES,$(POSITIVES)),10) \
		-numNeg $(call GET_NUM_SAMPLES,$(NEGATIVES)) \
		$(call GET_W_AND_H,$(call FIRST_FILE,$(POSITIVES))) \
		-numThreads $(NUM_THREADS) \
	   	-precalcValBufSize $(BUFFER_SIZE) \
		-precalcIdxBufSize $(BUFFER_SIZE) \
		|| rm -rf $@
	@cp $@/cascade.xml $@.xml

INTERMEDIATES=*_frames *_images fix_positives *s.txt samples samples.vec classifier
.PHONY: clean
clean:
	@echo " CLEAN $${PWD##*/}"
	@rm -rf $(INTERMEDIATES)
