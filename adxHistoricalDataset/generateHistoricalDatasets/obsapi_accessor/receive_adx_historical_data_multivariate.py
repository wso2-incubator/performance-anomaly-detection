# Obtain distinct obsId|version from ADX 30 days ago.
# For each application receive data and write to csv files.

from azure.kusto.data.helpers import dataframe_from_result_table
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder, ClientRequestProperties

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

# The ADX database name
db = "dev-observability"

# Output directory
multivariate_datasets_dir = "../experimentFolder/multivariateDatasets/"

def get_distinct_obsid_version():
    adx_query = """metrics
    | where timestamp > ago(30d)
    | distinct obsId, ['version']"""

    query_response = client.execute(db, adx_query)
    query_response_dataframe = (dataframe_from_result_table(query_response.primary_results[0])).fillna(0)
    return query_response_dataframe

def check_suitability_of_app(obsId,version):
    adx_query = """let uuidMetrics = metrics
    | where obsId == '%s' and version== '%s';
    let ServiceMetrics = uuidMetrics
       | where ['tags']['src.entry_point.resource'] == 'true';
    let CallerResponseMetrics = uuidMetrics
       | where ['tags']['action'] == 'respond' and ['tags']['connector_name'] == 'ballerina/http/Caller';
    let RequestsTotal = ServiceMetrics
       | where metricName == 'requests_total'
       | project value, timestamp, timeWindow=(toint(['tags']['timeWindow']))
       | summarize requests_total=sum(value) by timestamp, timeWindow;
    let ResponseTimeTotal = ServiceMetrics
       | where metricName == 'response_time_nanoseconds_total'
       | summarize response_time_total=(sum(value)) by timestamp;
    let AvgResponseTime = ResponseTimeTotal
       | join kind=leftouter (RequestsTotal) on timestamp
       | project timestamp, avg_response_time=((response_time_total / requests_total)/(1000*1000));
    let Throughput = RequestsTotal
       | project timestamp, throughput=((requests_total / timeWindow)*1000);
    let httpErrors = CallerResponseMetrics
       | summarize http_error_count=countif(isnotnull(['tags']['http.status_code_group']) and ['tags']['http.status_code_group'] != '2xx' and ['tags']['http.status_code_group'] != '1xx') by timestamp;
    let ballerinaErrors = ServiceMetrics
       | summarize ballerina_error_count=countif(['tags']['error'] == 'true') by timestamp;
    uuidMetrics
       | where metricName in ('requests_total', 'response_time_nanoseconds_total')
       | distinct timestamp
       | join kind=leftouter (AvgResponseTime) on timestamp
       | project-away timestamp1
       | join kind=leftouter (Throughput) on timestamp
       | project-away timestamp1
       | join kind=leftouter (httpErrors) on timestamp
       | project-away timestamp1
       | join kind=leftouter (ballerinaErrors) on timestamp
       | project-away timestamp1
       | order by timestamp asc;
    """%(obsId, version)
    query_response = client.execute(db, adx_query)
    query_response_dataframe = (dataframe_from_result_table(query_response.primary_results[0]))
    return query_response_dataframe

def get_observability_metrics(obsId,version):
    adx_query = """set truncationmaxsize=67108864;
    set truncationmaxrecords=500000;
    metrics
    | where obsId == '%s' and version== '%s';
    """%(obsId, version)
    query_response = client.execute(db, adx_query)
    query_response_dataframe = (dataframe_from_result_table(query_response.primary_results[0]))
    return query_response_dataframe

if __name__ == '__main__':
    final_df = get_distinct_obsid_version()
    final_df_dict = final_df.to_dict(orient='index')
    empty_df_count=0
    none_throughput_count=0
    insufficient_throughput_row_count=0
    for i in range(len(final_df_dict)):
        obsId = final_df_dict[i]['obsId']
        version = final_df_dict[i]['version']
        df = check_suitability_of_app(obsId, version)
        if df.empty:
            empty_df_count+=1
        else:
            if (df['throughput'].isnull().values.all()):
                none_throughput_count+=1
            else:
                if len(df)<10:
                    insufficient_throughput_row_count+=1
                else:
                    output_file_name = str(obsId + "|" + version)
                    print(output_file_name)
                    # Write processed metrics (throughput) to a separate directory
                    processed_output_file_name = str(obsId + "|" + version)
                    full_path_to_processed_output_file = multivariate_datasets_dir + str(output_file_name) + '.csv'
                    df.to_csv(full_path_to_processed_output_file, index=False)
                    # obtain observability metrics (However the queried data could exceed maximum file size limits)
                    # obs_df=get_observability_metrics(obsId, version)
                    # Write df to a csv identified by obsId|version
                    # full_path_to_output_file = original_metrics_dir + str(output_file_name) + '.csv'
                    # obs_df.to_csv(full_path_to_output_file, index=False)

    print("Number of applications is "+str(len(final_df_dict)))
    print("Number of applications without throughput data is "+str(empty_df_count))
    print("Number of applications with none values for the entire throughput column is "+str(none_throughput_count))
    print("Number of applications with less than 10 throughput rows is "+str(insufficient_throughput_row_count))
    print("Number of applications with historical data collected is "+str(len(final_df_dict)-(empty_df_count+none_throughput_count+insufficient_throughput_row_count)))