#!/usr/bin/python:!
import print_data_model
import re
    
# Typical file looks like:
example = '''
Video: video-1.mp4
Frame:  75% Correct
Frame: 100% Correct
Frame:   0% Correct
Frame:  25% Correct

... up to N Frames

Frame:   0% Correct
Frame:  25% Correct

Video: video-2.mp4
Frame:  75% Correct
Frame: 100% Correct
Frame:   0% Correct
Frame:  25% Correct

... up to N Frames

Frame:   0% Correct
Frame:  25% Correct

... up to N videos

Video: video-n.mp4
Frame:  75% Correct
Frame: 100% Correct
Frame:   0% Correct
Frame:  25% Correct

... up to N Frames

Frame:   0% Correct
Frame:  25% Correct
'''

class ResultMetrics(print_data_model.MetricContainer):
    # Data model class for Cache Simulator outputs

    def __init__(self, datastring):
        
        # Get array of videos
        videos = datastring.split('Video:')
        if len(videos) < 1:
            raise ValueError("No videos!")

        # Clean
        videos = map(lambda v: v.strip(), videos)
        videos = filter(lambda v: v is not '', videos)

        # For each array of videos, get array of frames
        videos = map(lambda v: v.split("Frame:"), videos)
        
        # Clean
        videos = map(lambda v: map(lambda l: l.strip(), v), videos)
        videos = map(lambda v: filter(lambda l: l is not '', v), videos)
        
        # For each frame, strip everything after the number (% Correct...)
        # NOTE: Might remove additional data, but that data shouldn't be there
        num_extract = lambda l: int(re.findall(r'\d+', l)[0])
        videos = map(lambda v: [v[0]] + map(lambda l: num_extract(l), v[1:]), videos)
        
        # For each video, extract the frame result data
        def data_extract(frames):
            results = dict()
            num_frames = len(frames)
            if num_frames == 0:
                raise ValueError("No frames in video!")
            results["num-frames"] = num_frames
            avg_score = sum(frames)/num_frames
            results["avg-score"] = int(avg_score)
            return results
        
        # Change to list of video name (remove extension), video data pairs
        videos = map(lambda v: [v[0], data_extract(v[1:])], videos)

        # Helper function for flattening list of lists
        flatten = lambda l: [item for sublist in l for item in sublist]

        # For each metric, append the video's name
        get_metric = lambda v: [v[0].replace('.mp4', '') + '-' + m for m in v[1].keys()]
        metric_list = map(get_metric, videos)
        metric_list = flatten(metric_list)
        
        # For each metric, append the video's name and capitolize
        clean_metric = lambda m: m.replace('-', ' ').\
                replace('num','number of').replace('avg','average').title()
        get_name = lambda v: [clean_metric(m) + ' ' + v[0] for m in v[1].keys()]
        title_list = map(get_name, videos)
        title_list = flatten(title_list)
        
        # For each metric, just return it's value
        get_data = lambda v: v[1].values()
        data_list = map(get_data, videos)
        data_list = flatten(data_list)
        
        # Total score (average over all videos)
        get_score = lambda v: v[1]["avg-score"]
        scores = map(get_score, videos)
        total_score = reduce(lambda s1, s2: s1 + s2, scores)
        total_score /= float(len(videos))
        
        metric_list.append('total-avg-score')
        title_list.append('Total Average Score')
        data_list.append(int(total_score))

        # Call superclass Container object to intialize data model
        print_data_model.MetricContainer.__init__(self, metric_list, title_list, data_list)
    
# Just define data model above and
# pass class to print_data_model module
if __name__ == '__main__':
    print_data_model.main(ResultMetrics, example)
