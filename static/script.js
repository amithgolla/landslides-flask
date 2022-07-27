var base64_str = "";

function previewFile() {
    const preview = document.querySelector('#before');
    const file = document.querySelector('input[type=file]').files[0];
    const reader = new FileReader();
  
    reader.addEventListener("load", function () {
      // convert image file to base64 string
      var paragraph = document.getElementById("beforetext");
        paragraph.textContent = "Before processing.";
      preview.src = reader.result;
      base64_str = reader.result;
      console.log(base64_str);
    }, false);
  
    if (file) {
      reader.readAsDataURL(file);
    }
    document.getElementById('processbtn').innerHTML = '<button onclick="afterclick1()">Click to process</button>'
    document.getElementById('gsibtn').innerHTML = '<button onclick="afterclick2()">Calculate GSI</button>'
  }

function getVals(d_strike, s_strike, d_dip, s_dip, f_angle){
  var str = d_strike.toString() + " " + s_strike.toString() + " " + d_dip.toString() + " " + s_dip.toString() + " " + f_angle.toString();
  console.log(str);
  getFailure(str);
}

function getFailure(str){
  fetch("/failure", {
     
    // Adding method type
    method: "POST",
     
    // Adding body or contents to send
    body: str,
})
 
// Converting to JSON
.then(response => response.text())
 
// Displaying results to console
.then(data => {
  console.log(data);
  //alert(data);
  document.getElementById("f_text").textContent = data;
});

}

function afterclick1(){
  fetch("/result", {
     
    // Adding method type
    method: "POST",
     
    // Adding body or contents to send
    body: base64_str,
})
 
// Converting to JSON
.then(response => response.json())
 
// Displaying results to console
.then(json => {
  console.log(json);
  const after_img = document.querySelector('#after');
  var processed_str = "data:image/jpeg;base64," + json['res_uri'];
  var array = json['linespacing'];
  var paragraph = document.getElementById("aftertext");
  paragraph.textContent = "After processing.";
  after_img.src = processed_str;
  document.getElementById("linespacingtext").textContent = 'Linespacing:\r\n' + array.join(", ");
  document.getElementById("rqdtext").textContent = 'Rock Quality Designation(RQD):\r\n' + json['rqd'];
});


}

function afterclick2(){
  fetch("/gsi", {
     
    // Adding method type
    method: "POST",
     
    // Adding body or contents to send
    body: base64_str,
})
 
// Converting to JSON
.then(response => response.text())
 
// Displaying results to console
.then(data => {
  console.log(data);

  document.getElementById("gsitext").textContent = 'Geological Strength Index(GSI):\r\n' + data;
});


}

function reloadpage(){
  window.location.reload();
}



//   function processImage(img) {
//     //console.log(img);
//     fetch("/result", {
//         method: "POST",
//         body: img
//     }).then(resp => resp.text())
//     .then(data => {

        
//         console.log(data);
//         const after_img = document.querySelector('#after');
//         var processed_str = "data:image/jpeg;base64," + data;
//         var paragraph = document.getElementById("aftertext");
//         paragraph.textContent += "After processing.";
//         after_img.src = processed_str;

        
//     });
// }





