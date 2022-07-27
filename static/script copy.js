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
    document.getElementById('processbtn').innerHTML = '<button onclick="afterclick()">Click to process</button>'
  }

async function afterclick(){
  const element = document.getElementById("processbtn");
  element.remove();
  async function processImage(img) {
    let response = await fetch("/result", {
      method: 'POST',
      body: img
    });
          var data = await response.text();
          const after_img = document.querySelector('#after');
          var processed_str = "data:image/jpeg;base64," + data;
          var paragraph = document.getElementById("aftertext");
          paragraph.textContent = "After processing.";
          after_img.src = processed_str;
  }
  await processImage(base64_str)
    .catch(e => {
      console.log('There has been a problem with your fetch operation: ' + e.message);
    });


    async function getlinespacing() {
      let response = await fetch("/linespacing", {
        method: 'GET',
      });
            var data = await response.text();
            var paragraph = document.getElementById("linespacingtext");
            paragraph.textContent = "Joint spacing: " + data;
            
    }
    await getlinespacing()
      .catch(e => {
        console.log('There has been a problem with your fetch operation: ' + e.message);
      });

      async function getrqd() {
        let response = await fetch("/rqd", {
          method: 'GET',
        });
              var data = await response.text();
              var paragraph = document.getElementById("rqdtext");
              paragraph.textContent = "RQD(Rock Quality designation): " + data;
              
      }
      await getrqd()
        .catch(e => {
          console.log('There has been a problem with your fetch operation: ' + e.message);
        });

      document.getElementById("reloadbtn").innerHTML = '<button onclick="reloadpage()">Process next image</button>';
  
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





