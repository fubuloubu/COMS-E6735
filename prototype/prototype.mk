# rule to run python scripts
.PHONY: %.pyrun
%.pyrun: %.py
	@echo "  RUN xiwi $<"
	@python -m py_compile $< # Compile to parse for errors
	@xiwi -T ./$< >/dev/null 2> $@ && cat $@ | awk '/Traceback/{flag=1}/Running exit commands/{flag=0}flag' && rm $@

# clean rule for prototype scripts
.PHONY: clean
clean:
	@echo " CLEAN prototype"
	@rm -f *.pyc
