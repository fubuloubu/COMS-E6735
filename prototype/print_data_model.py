import argparse
from tabulate import tabulate
import re
import os

# Required to invoke matplotlib
# without a graphical environment
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# Command used to print plots
from matplotlib2tikz import save as tikz_save

def transpose(a):
    return [[j[i] for j in a] for i in range(len(a[0]))]

class MetricContainer(object):
    
    # Helper class containing metric attributes
    class Metric:
        def __init__(self, title, data):
            self.title = title
            self.data = data

    # Initialize all the metrics we need
    # If called directly, this object will have nothing
    def __init__(self, metric_list, title_list, data_list):
        # dict class var of metric object names and instances
        self.metrics = {}
        for i, metric in enumerate(metric_list):
            self.metrics[metric] = self.Metric(title_list[i], data_list[i])
    
    # Utility function to return all metric names
    def get_metric_names(self):
        return list(self.metrics.keys())

    # Utility function to return all matching metric names
    def get_matching_names(self, match):
        return [s for s in self.get_metric_names() if re.search(match, s) is not None]
    
    # Return a metric object of a certain name
    def get_metric(self, metric_name):
        return self.metrics[metric_name]

    # Return the title for a metric
    def get_title(self, metric_name):
        return self.get_metric(metric_name).title

    # Return the data fore a metric
    def get_data(self, metric_name):
        return self.get_metric(metric_name).data
    
    # Utility function to return any special metric rules
    def get_special_metrics(self):
        return ['all']

    # Check if a globbing rule applies
    # Could be overriden if we wanted more globs available
    def check_glob(self, metric_name_list):
        # Supported glob pattern is * char
        matching = [s for s in metric_name_list if "*" in s]
        if metric_name_list == ['all']:
            # all keyword was found, return all keys
            return self.get_metric_names()
        elif matching is not None:
            # a Glob was found, for each one:
            for match in matching:
                i = metric_name_list.index(match)
                metric_name_list.pop(i)
                for metric in self.get_matching_names(match):
                    metric_name_list.insert(i, metric)
                    # increment index because we just inserted something
                    i = i + 1

            if len(metric_name_list) < 1:
                raise ValueError("Wildcard(s) %s didn't work" % ", ".join(matching))

            return metric_name_list
        else:
            return metric_name_list

    # Return a list of metrics
    def get_metric_list(self, metric_name_list):
        metric_name_list = self.check_glob(metric_name_list)
        return map(self.get_metric, metric_name_list)
        
    # Return all metric objects if no filter is applied, also allows overriding behavior
    def get_all_metrics(self):
        return get_metric_list(['all'])

    # Return titles for a list of metrics
    def get_title_list(self, metric_name_list):
        metric_name_list = self.check_glob(metric_name_list)
        return map(self.get_title, metric_name_list)
        
    # Return all data if no filter is applied, also allows overriding behavior
    def get_all_titles(self):
        return self.get_title_list(['all'])

    # Return a list of metric data
    def get_data_list(self, metric_name_list):
        metric_name_list = self.check_glob(metric_name_list)
        return map(self.get_data, metric_name_list)
        
    # Return all data if no filter is applied, also allows overriding behavior
    def get_all_data(self):
        return self.get_data_list(['all'])

    # Utility function to ensure two objects have the same metric names
    def check_metrics_match(self, another_instance):
        return self.metrics.viewkeys() == another_instance.metrics.viewkeys()

    # Utility function to ensure two objects hae the same titles
    def check_titles_match(self, another_instance):
        return self.check_metrics_match(another_instance) and \
            all(map(lambda x: self.get_title(x) == another_instance.get_title(x), \
                    self.metrics.keys()))

    # Comparision method for metric between two objects
    def compare_metric(self, metric_name, another_instance, compare_func=lambda x, y: x != y):
        baseline = self.get_data(metric_name)
        target = another_instance.get_data(metric_name)
        return compare_func(baseline, target)

    # Compare a set of metrics
    def compare_metric_list(self, metric_name_list, another_instance, compare_func=lambda x, y: x != y):
        metric_name_list = self.check_glob(metric_name_list)
        return map(lambda x: self.compare_metric(x, another_instance, compare_func), metric_name_list)
    
    # Compare all metrics
    def compare_all_metrics(self, another_instance, compare_func=lambda x, y: x != y):
        return self.compare_metric_list(['all'], another_instance, compare_func)

    # Print all data if no filter is applied, also allows overriding behavior
    def compare_config(self, another_instance, metric_name_list=None):
        if metric_name_list is None:
            return True
        else:
            return False

def compare(datamodel, fname_baseline, fname_target, fmt, selected_metrics, name_replace_list):
    # Compare two file data objects and return a table

    # Read data and initialize data models
    f = open(fname_baseline, 'r')
    baseline_data = datamodel(f.read())
    f = open(fname_target, 'r')
    target_data = datamodel(f.read())

    # Ensure matching configuration
    if not baseline_data.compare_config(target_data):
        raise AssertionError("Configuration not matching")
    
    # Compare performance improvement
    def percentage_improvement(baseline_val, target_val):
        return (baseline_val, target_val, (float(baseline_val) - float(target_val)) / float(baseline_val) * 100.0)
        
    if selected_metrics is None:
        data = baseline_data.compare_all_metrics(target_data, percentage_improvement)
        titles = baseline_data.get_all_titles() # titles should match since we checked config
    else:
        data = baseline_data.compare_metric_list(selected_metrics, target_data, percentage_improvement)
        titles = baseline_data.get_title_list(selected_metrics) # titles should match since we checked config

    # Return pretty table of performance data
    def create_row (title, data_item):
        baseline_val, target_val, percentage = data_item
        return [title, baseline_val, target_val, '{:+5.3f}%'.format(percentage)]

    rows = []
    for i, title in enumerate(titles):
        rows.append(create_row(title, data[i]))
    
    for (fstr, rstr) in name_replace_list:
        fname_baseline = fname_baseline.replace(fstr, rstr)
        fname_target = fname_target.replace(fstr, rstr)
    
    table = tabulate(rows, numalign="None", tablefmt=fmt, \
        headers=['Metric', fname_baseline, fname_target, 'Improvement [%]'])

    return table

def create_table(datamodel, target_list, fmt, selected_metrics, name_replace_list):
    data = []
    for i, target in enumerate(target_list):
        # Read data and initialize data model
        f = open(target, 'r')
        target_data = datamodel(f.read())
        
        if i == 0:
            # On first pass, create header list
            first = target_data
            if selected_metrics is None:
                data.append(["Result"] + target_data.get_all_titles())
            else:
                data.append(["Result"] + target_data.get_title_list(selected_metrics))
        else:
            # On subsequent passes, ensure they all have the same metrics and titles
            first.check_titles_match(target_data)

        # Perform find/replace on filename
        target_name = target
        for (fstr, rstr) in name_replace_list:
            target_name = target_name.replace(fstr, rstr)
        
        # Append all the data
        if selected_metrics is None:
            data.append([target_name] + target_data.get_all_data())
        else:
            data.append([target_name] + target_data.get_data_list(selected_metrics))
    
    data = transpose(data)
    hdrs = data.pop(0)
    return tabulate(data, tablefmt=fmt, numalign="None", headers=hdrs)

def graph(datamodel, target_list, selected_metrics, name_replace_list):
    
    xdata = []
    ydata = []
    for i, target in enumerate(target_list):
        # Read data and initialize data model
        f = open(target, 'r')
        target_data = datamodel(f.read())
        if i == 0:
            # On first pass, create header list
            first = target_data
            if selected_metrics is None:
                metrics = target_data.get_all_titles()
            else:
                metrics = target_data.get_title_list(selected_metrics)
        else:
            # On subsequent passes, ensure they all have the same metrics and titles
            first.check_titles_match(target_data)

        # Perform find/replace on filename
        target_name = target
        for (fstr, rstr) in name_replace_list:
            target_name = target_name.replace(fstr, rstr)
        xdata.append(target_name)

        # Append all the data
        if selected_metrics is None:
            ydata.append(target_data.get_all_data())
        else:
            ydata.append(target_data.get_data_list(selected_metrics))
    
    ydata = transpose(ydata)
    # Create graph
    fig = plt.figure()
    x = range(len(xdata))

    for i, metric in enumerate(metrics):
        plt.plot(x, ydata[i], 'o-', lw=4.0, label=metric)

    # Set temporary x-axis to text xdata
    plt.xticks(x, xdata)
    plt.legend(loc='lower right', bbox_to_anchor=(0.5, -0.05))
    
    # Write to a temporary file, 
    filename = "graph.tmp"
    tikz_save(filename)
    plt.close(fig)
    # get the string from that file,
    f = open(filename, 'r')
    graph_data = f.read()
    # remove the file,
    f.close()
    os.remove(filename)
    # then return the string
    return graph_data

def main(datamodel, example):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--format", metavar='FORMAT', default="simple",
        help="Sets output format (using options from package 'tabulate')")
    ap.add_argument("-m", "--metrics", nargs='+', metavar='METRIC', default=None,
        help="Filters the list of metrics to the desired set. " + \
                "The available options for this script include " + \
                "(normal): %s; (special): %s; any valid, matching regular expression" % \
                (" ".join(datamodel(example).get_metric_names()   ), \
                 " ".join(datamodel(example).get_special_metrics())) )
    # NOTE: Help for metrics includes parsing this script file, which is dangerous
    #       but allows us to easily return the names of all metrics available
    ap.add_argument("-r", "--replace", nargs='+', metavar='FIND REPLACE',
        help="Pair-ordered list of replacements to make in filenames\n" + \
            "e.g. FIND1 -> REPLACE1, FIND2 -> REPLACE2, etc.\n" + \
            "NOTE: use '+' char instead of '-' char")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("-t", "--tabulate", nargs='+', metavar='TARGET',
        help="Displays results data for each target in a table")
    g.add_argument("-c", "--compare", nargs=2, metavar=('BASELINE', 'TARGET'),
        help="Compares results data for target against baseline and presents a table")
    g.add_argument("-g", "--graph", nargs='+', metavar='TARGET',
        help="Compute a latex-friendly graph using TikZ for each target," +\
                "using target names for the x-axis and the selected metrics.")
    args = vars(ap.parse_args())
    
    raw_replace_list = args['replace']
    name_replace_list = []
    if raw_replace_list is not None:
        if (len(raw_replace_list) % 2):
            # Modulo isn't zero, therefore size of list is odd
            raise ValueError("replace list must be ordered pairs!\nlist: [\'" + \
                    '\', \''.join(raw_replace_list) + "\']")
        else:
            for i in range(0,len(raw_replace_list),2):
                # NOTE: user can substitute '+' char for '-' char to avoid argparse getting in the way
                name_replace_list.append((raw_replace_list[i].replace('+','-'), \
                        raw_replace_list[i+1].replace('+','-')))

    if args['tabulate'] is not None:
        print(create_table(datamodel, args['tabulate'], args['format'], args['metrics'], name_replace_list))
    elif args['compare'] is not None:
        # First is assumed to be the baseline, Second is assumed to be the target
        print(compare(datamodel, args['compare'][0], args['compare'][1], args['format'], args['metrics'], name_replace_list))
    elif args['graph'] is not None:
        print(graph(datamodel, args['graph'], args['metrics'], name_replace_list))

if __name__ == '__main__':
    raise UsageError("Cannot be used by itself")
