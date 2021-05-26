# Deep Bird Project


## Description

This project uses Heroku, Flask, AWS, and React Native to implement the ML model from the [2nd place winner](https://www.kaggle.com/c/birdsong-recognition/discussion/183269) of the Cornell Birdcall Identification Competition. User can run the React Native application on their iOS device to record a short birdcall clip, and the application will identify the bird and present the predicted bird species. 

https://user-images.githubusercontent.com/46701122/119613204-50024b00-be2f-11eb-8eb4-dda5814bcaba.mp4

## Getting Started
* The React Native code is located in the "DeepBird" folder on the react_native_app git branch. The Flask prediction app is located in the "BirdDetector" folder on the Singlemodel git branch. 


### Dependencies

* NodeJS
* Linux or MacOS
* pip
* Python3

### Set up you AWS S3
* Set up your S3 bucket through the following tutorial [AWS S3 Setup](https://habr.com/en/post/535054/)

### Installing

* Clone this project to your repository
  ```
  git clone https://github.com/yuchenc2/BirdDetector.git
  ```

* Install Expo and NodeJS
  ```
  npm install --global expo-cli
  ```

* Install requirements for React Native
  * Go to the DeepBird folder on the react_native_app git branch
  * Start a virtual environment
    ```
    python3 -m venv RNenv 
    source RNenv/bin/activate
    ```
  * NPM install all dependencies
    ```
    npm install
    ```
  * Create a folder called "auth" in the DeepBird folder and create a new file called "options.json" in there
  * Put your AWS authentication info in options.json like this:
    ```
    {
      "keyPrefix": "foldernameinbucket/",
      "bucket": "bucketname",
      "region": "your-aws-region",
      "accessKey": "youraccesskey",
      "secretKey": "yoursecretkey",
      "successActionStatus": 201
    }
    ```

* Install Expo Go on your phone

### Executing program
* Make sure your computer and your phone are on the same network
* Start Expo
  ```
  expo start
  ```
* Scan the QrCode and run the app. Enjoy!

## Software Stack

* After React Native app records the audio clip, the clip gets stored on an AWS S3 bucket. The app then does a fetch call with the name of the clip to the Flask app hosted on Heroku. The Flask app downloads the audio clip from the S3 bucket and runs the prediction. The result of the prediction gets returned as the fetch request response and presented on the React Native app.


## Heroku and Flask
* To push new code to Heroku, follow this to set up and deploy it [tutorial](https://stackabuse.com/deploying-a-flask-application-to-heroku/)

* If you want to run the Flask app locally on your computer instead of Heroku, follow these steps:
  * Go to the BirdDetector Folder and switch to "Singlemodel" git branch
  * Create a virtual environment and install the dependencies in requirements.txt
    ```
    python3 -m venv flaskenv 
    source flaskenv/bin/activate
    pip install -r requirements.txt
    ```
  * Run the flask app at http://127.0.0.1:5000 with the following command. The prediction code is now waiting for a GET request.
    ```
    python app.py
    ```
  * To do a simple GET request, do the following steps. 
    * You can see the recorded files in your S3 bucket by typing the following command. Copy the name of your file for the next step.
      ```
      aws s3 ls s3://deepbirdaudio/audioclips/
      ```
    * In a new terminal, type ``` python ```
    * Do a simple GET request by typing this
      ```
      import requests
      headers = {'file-name':'name of the file from the previous step'}
      response = requests.get("http://127.0.0.1:5000", headers=headers)
      ```

* Note: 
  * Heroku has a maximum slug size of 500MB, and this program size is currently at 406MB.
  * The recorded audio is stored as a WAV file (Required for the prediction algorithm), which is not supported for React Native's Expo recording function for Android.

## Version History

* 0.1
    * Initial Release

## Acknowledgments
* [ML Model](https://www.kaggle.com/c/birdsong-recognition/discussion/183269)
* [AWS S3 Setup](https://habr.com/en/post/535054/)
* [Flask with Heroku](https://stackabuse.com/deploying-a-flask-application-to-heroku/)
