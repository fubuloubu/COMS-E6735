# rule to run python scripts
.PHONY: %.pyrun
%.pyrun: %.py
	@echo "  RUN xiwi $<"
	@xiwi -F ./$<

# clean rule for prototype scripts
.PHONY: clean
clean:
	@echo " CLEAN prototype"
	@rm -f *.pyc
