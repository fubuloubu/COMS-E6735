# rule to run python scripts
.PHONY: %.pyrun
%.pyrun: %.py
	@echo "  RUN xiwi $<"
	@python -m py_compile $< # Compile to parse for errors
	@xiwi -F ./$< 2> $@ && rm $@ || cat $@

# clean rule for prototype scripts
.PHONY: clean
clean:
	@echo " CLEAN prototype"
	@rm -f *.pyc
