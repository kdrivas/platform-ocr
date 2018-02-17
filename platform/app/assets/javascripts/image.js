$(document).ready(function() {
  $('#upload_image').change(function(event) {
    var files = event.target.files;
    var image = files[0];
    // here's the file size
    console.log(image);
    var reader = new FileReader();
    reader.onload = function(file) {
      var img = new Image();
      img.src = file.target.result;
      console.log(file.target.result);
      img.id = "img_preview"

      var width = img.naturalWidth;
      var heigth = img.naturalHeight;
      console.log(width);
      console.log(heigth);


      if((width <= 800 && width >= 200) && (heigth <= 800 && heigth >= 200)){
        console.log("entree");
        $('#upload_image_view').html(img);
        $("#btn_submit_image").disabled = false;
      }
      else{
        $("#btn_submit_image").disabled = true;
      }
    }
    reader.readAsDataURL(image);
    console.log(files);
  });
});



