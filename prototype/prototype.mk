# rule to run python scripts
.PHONY: %.pyrun
%.pyrun: %.py
	@echo "  RUN xiwi $<"
	@python -m py_compile $< # Compile to parse for errors
	@xiwi -T ./$< >/dev/null 2> $@ && cat $@ | awk '/Traceback/{flag=1}/Running exit commands/{flag=0}flag' && rm $@

# rule for running canned input video and evaluating results
%.pytest: %.mp4 | %.py
	@echo " TEST $<"
	./$| $< 2> $*.pyrun
	@cat $*.pyrun | awk '/$|: Stats/{flag=1}/$|: Stats/{flag=0}flag' > $@
	@python -c "import $*; $*.verify('$@');"
	@rm $@

# clean rule for prototype scripts
.PHONY: clean
clean:
	@echo " CLEAN prototype"
	@rm -f *.pyc
	@rm -f *.pyrun
	@rm -f *.pytest
