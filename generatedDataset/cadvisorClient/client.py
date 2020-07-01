import requests
import os
import time
import csv
import json

full_container_id = os.popen('docker inspect --format="{{.Id}}" hwo1').read()
# Initialize the final_list
# final_list = []
final_timestamps_list = []
url = "http://127.0.0.1:8080/api/v2.1/stats/docker/"+str(full_container_id[:-1])
# Send the request
headers = {"content-type": "application/json", "Accept-Charset": "UTF-8"}
with open(os.path.join(os.pardir, "experimentFolder/system_metrics.csv"), 'a', newline='') as file:
    fieldnames = ['timestamp', 'cpu_usage', 'memory_usage']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
while True:
    r = requests.post(url, data={"sample":"data"}, headers=headers)
    data = r.json() #returns a dictionary
    main_key = '/docker/'+str(full_container_id[:-1])
    main_dict_value = data[main_key]
    # container_specs = main_dict_value['spec']
    container_stats = main_dict_value['stats']
    # Get the response and fill the final_list. Check for repeating timestamps. At the same time write to a csv
    # 16/06 - Parallely write i (the entire row) to another file
    for i in container_stats:
        perf_dictionary = {}
        perf_dictionary['timestamp'] = i['timestamp']
        # Total CPU usage in nanoseconds
        perf_dictionary['cpu_usage'] = i['cpu']['usage']['total']
        # Current memory usage, this includes all memory regardless of when it was accessed. Units: Bytes.
        # perf_dictionary['memory_usage'] = i['memory']['usage']
        # The amount of working set memory. Units: Bytes.
        perf_dictionary['memory_usage'] = i['memory']['working_set']
        final_list = []
        if perf_dictionary['timestamp'] not in final_timestamps_list:
            final_list.append(perf_dictionary)
            with open(os.path.join(os.pardir, "experimentFolder/system_metrics.csv"), 'a', newline='') as file:
                for item in final_list:
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writerow(item)
            with open(os.path.join(os.pardir, "experimentFolder/detailed_system_metrics.csv"), 'a') as file:
                file.write(json.dumps(i))
                file.write('\n')
            final_timestamps_list.append(i['timestamp'])
    # time.sleep(60)





