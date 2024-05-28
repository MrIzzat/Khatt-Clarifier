const dragArea = document.querySelector('.drag-area');
const dragText = document.querySelector('.header');

let button = document.querySelector('.button');
let input = document.querySelector('input');

let file;

button.onclick = () =>{
    input.click();
};

//When the browse button is clicked
input.addEventListener('change', function() {
    file = this.files[0];
    dragArea.classList.add('active');
    displayFile();

});

//When the file is inside the drag area:
dragArea.addEventListener('dragover',(event) => {
    event.preventDefault();
    dragText.textContent = "Release to Upload";
    dragArea.classList.add('active');

    //console.log('File is inside the drag area');
});


//When the file leaves the drag area
dragArea.addEventListener('dragleave', () => {
    dragText.textContent = "Drag & Drop";
    dragArea.classList.remove('active');
    
    // console.log('File left the drag area');
});

//When the file is dropped into the drag area
dragArea.addEventListener('drop', (event) =>{
    event.preventDefault();

    file = event.dataTransfer.files[0];
    console.log(file);

    displayFile();
   


});

function displayFile(){
    let fileType = file.type;
    //console.log(fileType);

    let validExtenstions = ['image/jpeg','image/jpg','image/png'];

    if(validExtenstions.includes(fileType)){
        let fileReader = new FileReader();
        let fileURL;
        fileReader.onload = () => {

            fileURL = fileReader.result;
            //let imgTag = `<img id="image" src="${fileURL}" alt="">`;
            //dragArea.innerHTML = imgTag;

            const fileInput = document.querySelector('#imageFile');

            // Get your file ready
            const myFileContent = [fileURL];
            const myFileName = 'test.png';
            const myFile = new File(myFileContent, myFileName);

            // Create a data transfer object. Similar to what you get from a `drop` event as `event.dataTransfer`
            const dataTransfer = new DataTransfer();

            // Add your file to the file list of the object
            dataTransfer.items.add(file);

            // Save the file list to a new variable
            const fileList = dataTransfer.files;

            // Set your input `files` to the file list
            fileInput.files = fileList;

            document.getElementById("ImageForm").requestSubmit();
        };
        fileReader.readAsDataURL(file);


    }else{
        alert('This file is not an Image')
        dragArea.classList.remove('active')
    }
        //console.log("The file is dropped in the drag area")
}