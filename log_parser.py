#!/usr/bin/env python2
from __future__ import division, unicode_literals, print_function
from six.moves import range, zip
from six import iteritems

import os, sys, re
from subprocess import check_output as co
from fnmatch import fnmatch

import numpy as np
import matplotlib.pyplot as plt
import datetime

from IPython.core.display import display, HTML
from IPython.core.display import display, HTML
from IPython.core.pylabtools import print_figure 
from matplotlib._pylab_helpers import Gcf

from base64 import b64encode


def str_timestamp(dt=None):

    if dt is None:
        dt = datetime.datetime.now()
    return "{day:02}-{month:02}-{year:04}_{hour_over_24:02}-{minutes:02}-{seconds:02}".format(
        day=dt.day, month=dt.month, year=dt.year, hour_over_24=dt.hour, minutes=dt.minute, seconds=dt.second)
    

def parse_timestamp(ts_str):
    ts_str = re.sub(r"\s", "", ts_str)
    assert re.match(r"\d{2}\-\d{2}\-\d{4}_\d{2}\-\d{2}\-\d{2}", ts_str), ts_str
    
    day = int(ts_str[:2])
    month = int(ts_str[3:5])
    year = int(ts_str[6:10])
    # Hour
    hours = int(ts_str[11:13])
    minutes = int(ts_str[14:16])
    seconds = int(ts_str[17:19])
    
    assert 1 <= day <= 31
    assert 1 <= month <= 12
    assert year == 2017
    
    assert 0 <= hours <= 23
    assert 0 <= minutes <= 59
    assert 0 <= seconds <= 59

    return datetime.datetime(year, month, day, hours, minutes, seconds)


def plot2data(fig):
    image_data = "data:image/png;base64,{}".format(b64encode(print_figure(fig)).decode("utf-8"))
    Gcf.destroy_fig(fig)
    return image_data


def plot2html(fig):
    return "<img src='{}'>".format(plot2data(fig))


def parse_date(file_name):   
    file_name = os.path.basename(file_name)
    assert file_name.endswith(".log"), file_name

    return parse_timestamp(file_name[:-4])


def file_names_and_datetimes_sorted_by_datetime(file_names):
    return reversed(sorted(({"datetime": parse_date(file_name), "file_name": file_name} for file_name in file_names), key=lambda x: x["datetime"]))


def parse_one_log(file_name_and_datetime, show_parsed_html=True, show_instance_html=True):
    file_name = file_name_and_datetime["file_name"]
    datetime_ = file_name_and_datetime["datetime"]

    assert os.path.exists(file_name), file_name

    html = ""
    html += "<hr class='file_seperator'>"
    html += "<div class='file'>"
    html += "<div class='file_title'>File '<span class='smaller'>{}</span>': </div>".format(file_name)
    html += "<hr class='instance_seperator'>"
    with open(file_name, "r") as f_:
        text = f_.read()
    
    if "Error" in text:
        print("Found an error in {}, skipping it.".format(file_name))
        return

    instances = text.split("--")
    parsed = {"args": None, "tr_inst": []}

    for instance_idx, instance in enumerate(instances):
        parsed_html = ""
        
        # Preprocess lines
        lines = [inst for inst in instance.split("\n") if not re.match(r"^\s+$", inst) and not inst == ""]
        lines_to_remove = set()
        for i, line in enumerate(lines):
            if line and len(line) > 300:
                print("Removed long ({}) line.".format(len(line)))
                # print("Removed long ({}) line. First c: '{}', Last c: '{}'".format(len(line), re.sub(r"\s+", " ", line[:200].replace("\r", "")), re.sub(r"\s+", " ", line[-200:].replace("\r", ""))))
                
                lines_to_remove.add(i)
        lines = [x for i, x in enumerate(lines) if not i in lines_to_remove]
        # If we are in an args instance:
        if "Args" in instance:
            instance_html = "<br>".join(lines)
            parsed_args = {}
            for line in lines:
                arg_match = re.match(r"\s*-\s*([\w_]+)\s*:\s*([\w_\-\./]+).*", line)
                if arg_match:
                    parsed_args[arg_match.group(1)] = arg_match.group(2)
            parsed["args"] = parsed_args
            parsed_html += "<div class='args'><h3>Args:</h3>{}</div>".format("<br>".join(["- <b>{}</b>: {}".format(k, v) for k, v in iteritems(parsed_args)]))


        elif "SVC" in instance:
            # If we are in a regular training instance
            instance_valid_score = None
            instance_training_score = None
            instance_C = None
            instance_gamma = None


            instance_html = "<br>".join(lines)

            
            
            for i, line in enumerate(lines):
                try:
                    valid_score_match = re.match(r"^\s*\-\s*Valid score:\s*([0-9\.\-Ee\+]+)\s*$", line)
                    training_score_match = re.match(r"^\s*\-\s*Training score:\s*([0-9\.\-Ee\+]+)\s*$", line)
                    C_match = re.match(r"^.*C=([0-9\.\-Ee\+]+).*$", line)
                    gamma_match = re.match(r".*\bgamma=([0-9\.\-Ee\+]+).*", line)

                    if valid_score_match:
                        instance_valid_score = float(valid_score_match.group(1))
                        parsed_html += "- <b>Valid score</b>: {:.3%}<br>".format(instance_valid_score)
                    elif training_score_match:
                        instance_training_score = float(training_score_match.group(1))
                        parsed_html += "- <b>Training score</b>: {:.3%}<br>".format(instance_training_score)
                    elif C_match and instance_C is None:
                        instance_C = float(C_match.group(1))
                        parsed_html += "- <b>C value</b>: {:.1e}<br>".format(instance_C)
                    elif gamma_match and instance_gamma is None:
                        instance_gamma = float(gamma_match.group(1))
                        parsed_html += "- <b>gamma value</b>: {:.1e}<br>".format(instance_gamma)
                except Exception as err:
                    print((i, line))
                    raise err

            parsed["tr_inst"].append(dict(
              v=instance_valid_score,
              t=instance_training_score,
              c=instance_C,
              g=instance_gamma,
              ))

        else:
            instance_html = "<br>".join(lines)


        if parsed_html == "":
            parsed_html += "[No parsed data]"
        parsed_html = """
            <div class='parsed'><h4>Parsed data:</h4>{}</div>
            """.format(parsed_html)


        if instance_html == "":
            instance_html += "[No original otuput]"
        instance_html = """
            <div class='original_output'><h4>Original output:</h4>{}</div>
            """.format(instance_html)

        html += "<div class='instance'>{}{}</div>".format(parsed_html if show_parsed_html else "", instance_html if show_instance_html else "")

        if instance is not instances[-1]:
            html += "<hr class='instance_seperator'>"




    html += "</div>"
    return html, parsed

def analyse_logs(path, show_parsed_html=True, show_instance_html=True):
    assert os.path.exists(path), path
    assert os.path.isdir(path), path

    files = [os.path.join(path, file_) for file_ in os.listdir(path) if fnmatch(file_, "*.log")]
    base_html = """
    <style type='text/css'>
    div.root {
        text-align: left;
    }

    div.file_title {
        font-size: 2em;
        margin-top: 1em;
        margin-bottom: 1em;
    }

    .smaller { 
        font-size: 0.9em;
    }
    .parsed {
        margin-bottom: 15px;
    }

    hr {
        border: 0; 
        border-top: 1px solid;
        display: block;
        margin-left: 0;
        margin-top: 60 px;
        margin-bottom: 60 px;
    }

    hr.file_seperator {
        width: 80%;
        border-color: #333;
        border-top: 2px;
    }

    hr.instance_seperator {
        width:50%;
        border-color: #AAA;
    }


    </style>
    """

    assert files, files
    # print(files)
    filtered = []
    for file_ in file_names_and_datetimes_sorted_by_datetime(files):
        parsed = parse_one_log(file_, show_parsed_html, show_instance_html)
        if parsed:
            filtered.append(parsed)

    per_file_html, per_file_parsed = zip(*filtered)
    assert per_file_html
    all_html = base_html + "<div class='root'>" + "\n".join(per_file_html)  + "</div>" 
    
    display(HTML(all_html))

    for per_file in per_file_parsed:
        parsed_no_nones = [x for x in per_file[0] if x["v"] is not None]
        for p in sorted(parsed_no_nones, key=lambda x: -x["v"]):
            # print("----")
            for k, v in iteritems(p):
                #print("- {}: {}".format(k, v))
                pass

        smallest_good_g = None
        biggest_good_g  = None
        smallest_good_c = None
        biggest_good_c  = None

        smallest_g = None
        biggest_g  = None
        smallest_c = None
        biggest_c  = None
        
        for p in parsed_no_nones:
            if p["v"] > .7:
                if smallest_good_g is None or p["g"] < smallest_good_g:
                    smallest_good_g = p["g"]
                if smallest_good_c is None or p["c"] < smallest_good_c:
                    smallest_good_c = p["c"]
                if biggest_good_g is None or p["g"] > biggest_good_g:
                    biggest_good_g = p["g"]
                if biggest_good_c is None or p["c"] > biggest_good_c:
                    biggest_good_c = p["c"]

            if smallest_g is None or p["g"] < smallest_g:
                smallest_g = p["g"]
            if smallest_c is None or p["c"] < smallest_c:
                smallest_c = p["c"]
            if biggest_g is None or p["g"] > biggest_g:
                biggest_g = p["g"]
            if biggest_c is None or p["c"] > biggest_c:
                biggest_c = p["c"]

        print("smallest_good_g: {}".format(smallest_good_g))
        print("biggest_good_g:  {}".format(biggest_good_g))
        print("smallest_good_c: {}".format(smallest_good_c))
        print("biggest_good_c:  {}".format(biggest_good_c))

        for p in parsed_no_nones:
            assert p["v"] < .7 or (smallest_good_g <= p["g"] <= biggest_good_g), ("g", p["v"], p["g"])
            assert p["v"] < .7 or (smallest_good_c <= p["c"] <= biggest_good_c), ("c", p["v"], p["c"])

        print("smallest_g: {}".format(smallest_g))
        print("biggest_g:  {}".format(biggest_g))
        print("smallest_c: {}".format(smallest_c))
        print("biggest_c:  {}".format(biggest_c))
        print("--------")

    
    

