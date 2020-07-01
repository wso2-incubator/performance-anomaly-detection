import ballerina/http;
import ballerina/log;

// Client endpoint to communicate with Airline reservation service
http:Client airlineReservationEP = new("http://192.168.1.3:7278");
// Experiment IPs
// http:Client airlineReservationEP = new("http://192.168.32.8:7278");

// // Client endpoint to communicate with Hotel reservation service
http:Client hotelReservationEP = new("http://192.168.1.4:6268");
// Experiment IPs
// http:Client hotelReservationEP = new("http://192.168.32.8:6268");

// // Client endpoint to communicate with Car rental service
http:Client carRentalEP = new("http://192.168.1.5:5258");
// Experiment IPs
// http:Client carRentalEP = new("http://192.168.32.8:5258");

// Travel agency service to arrange a complete tour for a user
@http:ServiceConfig {basePath:"/travel"}
service travelAgencyService on new http:Listener(9298) {
    
    // Resource to arrange a tour
    @http:ResourceConfig {methods:["POST"], consumes:["application/json"],
        produces:["application/json"]}
    resource function arrangeTour (http:Caller caller, http:Request inRequest) returns error? {
        http:Response outResponse = new;
        map<json> inReqPayload = {};

        // JSON payload format for an HTTP out request.
        map<json> outReqPayload = {"FirstName": "", "LastName": "", "ArrivalDate": "", "DepartureDate": "", "Preference": ""};

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
        outReqPayload["ArrivalDate"] = inReqPayload["ArrivalDate"];
        outReqPayload["DepartureDate"] = inReqPayload["DepartureDate"];
        json | error airlinePreference = inReqPayload.Preference.Airline;
        json | error hotelPreference = inReqPayload.Preference.Accommodation;
        json | error carPreference = inReqPayload.Preference.Car;

        // If payload parsing fails, send a "Bad Request" message as the response
        if (outReqPayload.Name is () || outReqPayload.ArrivalDate is () || outReqPayload.DepartureDate is () ||
        airlinePreference is error || hotelPreference is error || carPreference is error) {
            outResponse.statusCode = 400;
            outResponse.setJsonPayload({"Message": "Bad Request - Invalid Payload"});
            var result = caller->respond(outResponse);
            if (result is error) {
                log:printError(result.message(), err = result);
            }            
            return;
        }

        // Reserve airline ticket for the user by calling Airline reservation service
        // construct the payload
        map<json> outReqPayloadAirline = outReqPayload.clone();
        outReqPayloadAirline["Preference"] = <json>airlinePreference;

        // Send a post request to airlineReservationService with appropriate payload and get response
        http:Response inResAirline = check airlineReservationEP->post("/reserve/airline", <@untainted>outReqPayloadAirline);

        // Get the reservation status
        var airlineResPayload = check <@untainted>inResAirline.getJsonPayload();
        string airlineStatus = airlineResPayload.Message.toString();
        // If reservation status is negative, send a failure response to user
        if (airlineStatus != "Success") {
            outResponse.statusCode = 500;
            outResponse.setJsonPayload({
                "Message": "Failed to reserve airline! " +
                "Provide a valid 'Preference' for 'Airline' and try again"
            });
            var result = caller->respond(outResponse);
            if (result is error) {
                log:printError(result.message(), err = result);
            }
            return;
        }

        // Reserve hotel room for the user by calling Hotel reservation service
        // construct the payload
        map<json> outReqPayloadHotel = outReqPayload.clone();
        outReqPayloadHotel["Preference"] = <json>hotelPreference;

        // Send a post request to hotelReservationService with appropriate payload and get response
        http:Response inResHotel = check hotelReservationEP->post("/reserve/hotel", <@untainted>outReqPayloadHotel);

        // Get the reservation status
        var hotelResPayload = check <@untainted>inResHotel.getJsonPayload();
        string hotelStatus = hotelResPayload.Message.toString();
        // If reservation status is negative, send a failure response to user
        if (hotelStatus != "Success") {
            outResponse.statusCode = 500;
            outResponse.setJsonPayload({
                "Message": "Failed to reserve hotel! " +
                "Provide a valid 'Preference' for 'Accommodation' and try again"
            });
            var result = caller->respond(outResponse);
            if (result is error) {
                log:printError(result.message(), err = result);
            }
            return;
        }

        // Renting car for the user by calling Car rental service
        // construct the payload
        map<json> outReqPayloadCar = outReqPayload.clone();
        outReqPayloadCar["Preference"] = <json>carPreference;

        // Send a post request to carRentalService with appropriate payload and get response
        http:Response inResCar = check carRentalEP->post("/reserve/car", <@untainted>outReqPayloadCar);

        // Get the rental status
        var carResPayload = check <@untainted>inResCar.getJsonPayload();
        string carRentalStatus = carResPayload.Message.toString();
        // If rental status is negative, send a failure response to user
        if (carRentalStatus != "Success") {
            outResponse.statusCode = 500;
            outResponse.setJsonPayload({
                "Message": "Failed to rent car! " +
                "Provide a valid 'Preference' for 'Car' and try again"
            });
            var result = caller->respond(outResponse);
            if (result is error) {
                log:printError(result.message(), err = result);
            }
            return;
        }

        // If all three services response positive status, send a successful message to the user
        outResponse.setJsonPayload({"Message": "Congratulations! Your journey is ready!!"});
        var result = caller->respond(outResponse);
        if (result is error) {
            log:printError(result.message(), err = result);
        }    
    }
}