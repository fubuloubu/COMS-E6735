# rule to run python scripts
.PHONY: %.pyrun
%.pyrun: %.py
	xiwi -F ./$<

# clean rule for prototype scripts
.PHONY: clean
clean:
	@echo " CLEAN prototype"
	@rm -f *.pyc
