# If we have other specific rules, include them here
-include *.mk

# Status rule to check how much we have left
# Uses TODO and DONE flags in files
.PHONY: status
status:
	@echo "STATUS $${PWD##*/}/*.*"
	@\
for file in `ls *.* | egrep -v "(template)"`; do \
	todo_togo=$$(grep -c TODO $$file); \
	todo_done=$$(grep -c DONE $$file); \
	total=$$(expr $$todo_done + $$todo_togo); \
	if [ $$total -ne 0 ]; then \
		percent=$$(expr 100 \* $$todo_done / $$total); \
		echo "$$file: $$percent% complete ($$todo_done DONE of $$total TODO items)"; \
	fi \
done;
