function predictDisaster() {
    const disasterType = document.getElementById("disaster-type").value;
    const region = document.getElementById("region").value;
    const resultDiv = document.getElementById("result");
  
    if (!region) {
      resultDiv.textContent = "Please enter a region for analysis.";
      return;
    }
  
    let prediction = `Analyzing ${Select Month} risk for ${region}... `;
  
    switch (Select Month) {
      case "january":
        prediction += "NO Data avilable.";
        break;
      case "febuary":
        prediction += "No Data avilable.";
        break;
        case "march":
          prediction += "No Data avilable.";
          break;
          case "april":
        prediction += "No Data avilable.";
        break;
        case "may":
        prediction += "No Data avilable.";
        break;
        case "":
        prediction += "No Data avilable.";
        break;
        case "":
        prediction += "No Data avilable.";
        break;
        case "":
        prediction += "No Data avilable.";
        break;
        case "":
        prediction += "No Data avilable.";
        break;
        case "":
        prediction += "No Data avilable.";
        break;
        case "":
        prediction += "No Data avilable.";
        break;
        case "":
        prediction += "No Data avilable.";
        break;
        case "":
        prediction += "No Data avilable.";
        break;
      default:
        prediction += "No data available.";
    }
  
    resultDiv.textContent = prediction;
  }
  
  function showDisasterHistory(location) {
    const mapResultDiv = document.getElementById("map-result");
  
    let history;
    switch (location) {
      case "Mumbai":
        history = "Mumbai has faced severe flooding in 2005, and moderate earthquakes.";
        break;
      case "Delhi":
        history = "Delhi has moderate earthquake risks and occasional flooding.";
        break;
      case "Chennai":
        history = "Chennai experienced massive floods in 2015.";
        break;
      case "Kolkata":
        history = "Kolkata is prone to flooding and cyclones.";
        break;
      default:
        history = "No data available for this location.";
    }
  
    mapResultDiv.textContent = `Disaster history for ${location}: ${history}`;
  }
  