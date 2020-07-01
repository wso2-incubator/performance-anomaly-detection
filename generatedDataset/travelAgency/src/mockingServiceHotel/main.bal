import ballerina/io;
import ballerina/http;
import ballerina/log;
import ballerina/mysql;

string dbUser = "root";
string dbPassword = "test@123";

mysql:Client mysqlClient = check new (host= "192.168.1.7", port= 3306, user = dbUser, password = dbPassword, database = "hotel");
// Experiment IPs
// mysql:Client mysqlClient = check new (host= "192.168.32.4", port= 3301, user = dbUser, password = dbPassword, database = "hotel");

function readOperation(mysql:Client mysqlClient, map<json> input) returns int{
    stream<record{}, error> resultStream =
        mysqlClient->query("Select count(*) as Total from reservations where first_name='" +
            input.FirstName.toString() +
            "' and last_name='" +
            input.LastName.toString() +
            "'");
    record {|record {} value;|}|error? result = resultStream.next();
    error? e = resultStream.close();
    if (result is record {|record {} value;|}) {
        if result.value["Total"]==1{
            return 1;
        }else{
            return 0;
        }
    } else if (result is error) {
        io:println("Next operation on the stream failed!", result);
        return 0;
    } else {
        io:println("Query result is empty");
        return 0;
    }
}

// Hotel mocking service to check accomodation related details of a tour for a user
@http:ServiceConfig {basePath:"/reserve"}
service reserveItems on new http:Listener(6268) {
    // Resource to arrange accomodation related details of a tour
    @http:ResourceConfig {methods:["POST"], consumes:["application/json"],
        produces:["application/json"]}
    resource function hotel (http:Caller caller, http:Request inRequest) returns error? {
        log:printInfo("reserving hotel...");
        http:Response outResponse = new;
        map<json> inReqPayload = {};

        // JSON payload format to be sent for querying.
        map<json> outReqPayload = {"FirstName": "", "LastName": ""};
        // Try parsing the JSON payload from the user request
        var payload = inRequest.getJsonPayload();
        if (payload is map<json>) {
            // Valid JSON payload
            inReqPayload = payload;
        } else {
            // NOT a valid JSON payload
            outResponse.statusCode = 400;
            outResponse.setJsonPayload({"Message": "Invalid payload - Not a valid JSON payload"});
            var result = caller->respond(outResponse);
            if (result is error) {
                log:printError(result.message(), err = result);
            }
            return;
        }
        
        outReqPayload["FirstName"] = inReqPayload["FirstName"];
        outReqPayload["LastName"] = inReqPayload["LastName"];
        json | error hotelPreference = inReqPayload.Preference;
        
        // If payload parsing fails, send a "Bad Request" message as the response
        if (outReqPayload.FirstName is () || outReqPayload.LastName is () || hotelPreference is error) {
            outResponse.statusCode = 400;
            outResponse.setJsonPayload({"Message": "Bad Request - Invalid Payload"});
            var result = caller->respond(outResponse);
            if (result is error) {
                log:printError(result.message(), err = result);
            }            
            return;
        }
        
        // Reserve accomodation for the user by calling Hotel reservation service
        // Get the reservation status
        int readAck = readOperation(mysqlClient, <@untainted>  outReqPayload);
        if (readAck == 1) {
            io:println("Queried from the database successfully!");
            outResponse.statusCode = 200;
            outResponse.setJsonPayload({"Message": "Success"});
            var result = caller->respond(outResponse);
            if (result is error) {
                log:printError(result.message(), err = result);
            }
        } else {
            io:println("Data querying failed!");
            outResponse.statusCode = 500;
            outResponse.setJsonPayload({"Message": "Failed to reserve hotel! " +
                "Provide a valid 'Preference' for 'Hotel' and try again"});
            var result = caller->respond(outResponse);
            if (result is error) {
                log:printError(result.message(), err = result);
            }
        }
        return;
    }
}