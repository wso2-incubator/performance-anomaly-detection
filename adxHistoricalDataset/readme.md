* Both supervised and unsupervised (modified K-means) models are trained on YWS5 data set.
* Historical data collected from ADX metrics table for a period of 30 days, has been manually labeled.
    * 3 people independently labeled, the historical data.
    * During the unification process, only those data points which were marked by at least 2 people were labeled as 
    anomalous in the final data set.
* Detect anomalies in the ADX historical data set using both methods.

**Execution steps**

* generateHistoricalDatasets
    * obsapi_accessor/receive_adx_historical_data_multivariate.py - To retrieve 30 days old historical data from ADX
        * Output folder - experimentFolder/multivariateDatasets
    * visualizeData/visualizeDatasetsMultivariate.py - To visualize throughput, latency, ballerina error count and http error count as subplots in the main plot
        * Output folder - visualizeData/visualisationPlots
    * visualizeData/manualLabeling - 
        * Consists of manually labeled datasets by 3 people; Duneesha, IsuruRuh, Nilushan 
        * Unified_labels (atleast 2/3 of votes for anomalous points)
    * labelDatasets.py and doubleCheckLabeling.py - Used to label the multivariate datasets and double check labeling.
        * Destination folder - experimentFolder/labeledDatasets

* anomalyDetectors - Test the anomalyDetectors; supervised and unsupervised against ADX historical labeled data in experimentFolder/labeledDatasets

**Following are the evaluation results**

* Unsupervised Model (Modified K-means)

    * Relaxed evaluation conditions
        * Accuracy - 0.592
        * Precision - 0.007  
        * Recall - 0.468      
        * AUC - 0.531        
        * F1 - 0.014
        * Confusion Matrix - 
        [[6123 4199]
        [  34   30]]
                             
    * Non-relaxed evaluation conditions
        * Accuracy - 0.579
        * Precision - 0.005  
        * Recall - 0.344      
        * AUC - 0.462       
        * F1 - 0.01
        * Confusion Matrix - 
        [[5991 4331]
        [  42   22]]
        
* Supervised Model 

    * Relaxed evaluation conditions
        * Accuracy - 0.99
        * Precision - 0.08   
        * Recall - 0.08      
        * AUC - 0.54        
        * F1 - 0.08
        * Confusion Matrix - 
        [[10261    61]
        [   59     5]]
                             
    * Non-relaxed evaluation conditions
        * Accuracy - 0.99
        * Precision - 0.05
        * Recall - 0.06      
        * AUC - 0.53       
        * F1 - 0.06
        * Confusion Matrix - 
        [[10245    77]
        [   60     4]]