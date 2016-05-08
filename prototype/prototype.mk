# All the targets we have
TARGETS=locate.results handmodel.results guitarmodel.results musicmodel.results

PYTHON=python

SHOW=xiwi -T
IGNORE=/dev/null
EXECUTE=./$(1) $(2) 2>$(4) > $(3)

AWK=awk
ERROR_FILTER='/Traceback/{flag=1}/Running exit commands/{flag=0}flag'
CHECK_ERROR=$(AWK) $(ERROR_FILTER)

# rule to run python scripts
.PHONY: %.pyrun
%.pyrun: %.py
	@echo "  RUN $<"
	@$(SHOW) $(call EXECUTE,$<,,$(IGNORE),$@)
	@cat $@ | $(CHECK_ERROR)
	@rm $@

STATS_FILTER='/$(1): Stats/{flag=1}/$(1): Stats/{flag=0}flag'
FILTER_STATS=$(AWK) $(STATS_FILTER)
#@cat $$@ | $(call FILTER_STATS,$$<) > $$@

# Cat all of them together
VIDEO_FILES=$(shell ls *.mp4)

# Rule for running canned input video and evaluating results
define TEST_VIDEO_RULES
%.$(1).pytest: %.py
	@echo " TEST $$< > $(1)"
	@$(call EXECUTE,$$<,$(1),$(IGNORE),$$@)

%.$(1).results: %.$(1).pytest
	@echo "  GEN $$@"
	@$(PYTHON) -c "import $$*; $$*.verify('$$<');" > $$@
endef
# Generate a rule for each video
$(foreach video,$(VIDEO_FILES),$(eval $(call TEST_VIDEO_RULES,$(video))))

# Collate all the video results files into one
%.results: $(foreach video,$(VIDEO_FILES),%.$(video).results ) | %.py
	@echo "  GEN $@"
	@touch $@
	@$(foreach rslt,$^,cat $(rslt) >> $@;)

# Collate all the results files into one table
METRICS_SCRIPT=results_evaluation.py
METRICS_OPTIONS  = -m $(foreach video,$(VIDEO_FILES),$(subst .mp4,-avg-score,$(video)))
METRICS_OPTIONS += total-avg-score
METRICS_OPTIONS += -f latex_booktabs
GET_METRICS=$(PYTHON) $(METRICS_SCRIPT) $(METRICS_OPTIONS)
results.tex: $(foreach target,$(TARGETS),$(subst py,results,$(target)))
	@echo "  GEN $@"
	@$(GET_METRICS) -t $^

# clean rule for prototype scripts
.PHONY: clean
clean:
	@echo "CLEAN prototype"
	@rm -f *.pyc
	@rm -f *.pyrun
	@rm -f *.pytest
	@rm -f *.results
