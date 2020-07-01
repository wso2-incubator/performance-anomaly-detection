import os
from azure.kusto.data.helpers import dataframe_from_result_table
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder

from generatedDataset.adxClient.adxConfig import Config

######################################################
##                        AUTH                      ##
######################################################

cluster = Config.cluster
client_id = Config.client_id
client_secret = Config.client_secret
authority_id = Config.authority_id

kcsb = KustoConnectionStringBuilder.with_aad_application_key_authentication(cluster, client_id, client_secret, authority_id)
client = KustoClient(kcsb)

######################################################
##                       QUERY                      ##
######################################################

# Observability ID
UUID = 'obsid_338d0e82-98dd-467c-8a15-5443f50e9e4a/v-60b28f6a-8509-4922-988f-d52029566468'

# The ADX database name
db = "dev-observability"

# Query to get all required metrics
def get_observability_metrics():
    adx_query = """let uuidMetrics = metrics
    | where obsId == '%s' and version== '%s';
    let inMetrics = uuidMetrics
    | where ['tags']['src.entry_point.resource'] == 'true';
    let outMetrics = uuidMetrics
    | where ['tags']['connector_name'] == 'ballerina/http/Client';
    let inRequestsTotal = inMetrics
    | where metricName == 'requests_total'
    | project value, timestamp, timeWindow=(toint(['tags']['timeWindow']))
    | summarize in_requests_total=sum(value) by timestamp, timeWindow;
    let inResponseTimeTotal = inMetrics
    | where metricName == 'response_time_nanoseconds_total'
    | summarize in_response_time_total=(sum(value)) by timestamp;
    let inIPR = inMetrics
    | where metricName == 'inprogress_requests'
    | summarize in_ipr=sum(value) by timestamp;
    let inAvgResponseTime = inResponseTimeTotal
    | join kind=leftouter (inRequestsTotal) on timestamp
    | project timestamp, in_avg_response_time=((in_response_time_total / in_requests_total)/(1000*1000));
    let inThroughput = inRequestsTotal
    | project timestamp, in_throughput=((in_requests_total / timeWindow)*1000);
    let outRequestsTotal = outMetrics
    | where metricName == 'requests_total'
    | summarize out_requests_total=sum(value) by timestamp, timeWindow=(toint(['tags']['timeWindow'])), position=tostring(['tags']['src.position']);
    let outResponseTimeTotal = outMetrics
    | where metricName == 'response_time_nanoseconds_total'
    | summarize out_response_time_total=(sum(value)) by timestamp, position=tostring(['tags']['src.position']);
    let outIPR = outMetrics
    | where metricName == 'inprogress_requests'
    | summarize out_ipr=sum(value) by timestamp, position=strcat('out_inprogress_requests_', tostring(['tags']['src.position']))
    | evaluate pivot(position, sum(out_ipr));
    let outAvgResponseTime = outResponseTimeTotal
    | join kind=leftouter (outRequestsTotal) on timestamp, position
    | project timestamp, out_avg_response_time=((out_response_time_total / out_requests_total)/(1000*1000)), position=strcat('out_avg_response_time_', position)
    | evaluate pivot(position, sum(out_avg_response_time));
    let outThroughput = outRequestsTotal
    | project timestamp, out_throughput=((out_requests_total / timeWindow)*1000), position=strcat('out_throughput_', position)
    | evaluate pivot(position, sum(out_throughput));
    uuidMetrics
    | where metricName in ('inprogress_requests', 'requests_total', 'response_time_nanoseconds_total')
    | distinct timestamp
    | join kind=leftouter (inAvgResponseTime) on timestamp
    | project-away timestamp1
    | join kind=leftouter (inThroughput) on timestamp
    | project-away timestamp1
    | join kind=leftouter (inIPR) on timestamp
    | project-away timestamp1
    | join kind=leftouter (outAvgResponseTime) on timestamp
    | project-away timestamp1
    | join kind=leftouter (outThroughput) on timestamp
    | project-away timestamp1
    | join kind=leftouter (outIPR) on timestamp
    | project-away timestamp1
    | order by timestamp asc;
    """%(UUID.split('/')[0], UUID.split('/')[1])

    query_response = client.execute(db, adx_query)
    query_response_dataframe = (dataframe_from_result_table(query_response.primary_results[0])).fillna(0)
    return query_response_dataframe

final_df = get_observability_metrics()

final_df.to_csv(os.path.join(os.pardir, "experimentFolder/application_metrics.csv"), index=False)
print(final_df)