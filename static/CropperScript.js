const image = document.getElementById('image');

const cropper = new Cropper(image);


document.querySelector('#btn-crop').addEventListener('click', function(){
    var croppedImage = cropper.getCroppedCanvas().toDataURL("image/png");
    document.getElementById('output').src = croppedImage;
    document.querySelector(".cropped-container").style.display = 'flex';
});



document.querySelector('#submitImage').addEventListener('click', function(){

    var blob =  cropper.getCroppedCanvas().toDataURL();
    var croppedImage =  dataURItoBlob(blob);

    console.log(croppedImage);

    const fileInput = document.querySelector('#imageFile');

    // Get your file ready
    const myFileContent = [croppedImage];
    const myFileName = 'test2.png';
    const myFile = new File(myFileContent, myFileName);

    // Create a data transfer object. Similar to what you get from a `drop` event as `event.dataTransfer`
    const dataTransfer = new DataTransfer();

    // Add your file to the file list of the object
    dataTransfer.items.add(myFile);

    // Save the file list to a new variable
    const fileList = dataTransfer.files;

    // Set your input `files` to the file list
    fileInput.files = fileList;

    document.getElementById("ImageForm").requestSubmit();
});

function dataURItoBlob(dataURI) {
    // convert base64/URLEncoded data component to raw binary data held in a string
    var byteString;
    if (dataURI.split(',')[0].indexOf('base64') >= 0)
        byteString = atob(dataURI.split(',')[1]);
    else
        byteString = unescape(dataURI.split(',')[1]);

    // separate out the mime component
    var mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];

    // write the bytes of the string to a typed array
    var ia = new Uint8Array(byteString.length);
    for (var i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }

    return new Blob([ia], {type:mimeString});
}

function loading(){

    var loading = document.getElementById('loading');
    var content = document.getElementById('content');
    var body = document.body;


    body.className = "body2";
    loading.style.display   = 'flex';
    content.style.display = 'none';

 }