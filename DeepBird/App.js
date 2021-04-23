import { StatusBar } from 'expo-status-bar';
import React, { Component } from 'react';
import { Button, StyleSheet, Text, View } from 'react-native';
import Icon from 'react-native-vector-icons/FontAwesome';
import { RNS3 } from 'react-native-aws3';
import { Audio } from 'expo-av';

export default function App() {
  const [recording, setRecording] = React.useState();
  const [prediction, setPrediction] = React.useState("");
  const options = require('./auth/options.json');

  function generateUUID() {
    var d = new Date().getTime();
    var uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = (d + Math.random()*16)%16 | 0;
        d = Math.floor(d/16);
        return (c=='x' ? r : (r&0x3|0x8)).toString(16);
    });
    return uuid; 
  };

  async function startRecording() {
    setPrediction("")
    try {
      console.log('Requesting permissions..');
      await Audio.requestPermissionsAsync();
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      }); 
      console.log('Starting recording..');
      const recording = new Audio.Recording();
      await recording.prepareToRecordAsync(Audio.RECORDING_OPTIONS_PRESET_HIGH_QUALITY);
      await recording.startAsync(); 
      setRecording(recording);
      console.log('Recording started');
    } catch (err) {
      console.error('Failed to start recording', err);
    }
  }

  async function stopRecording() {
    console.log('Stopping recording..');
    setRecording(undefined);
    await recording.stopAndUnloadAsync();
    const uri = recording.getURI(); 
    console.log('Recording stopped and stored at', uri);
    predictBird();
  }

  async function predictBird() {
    console.log('Uploading audio file..')
    const file = {
      uri: recording.getURI(),
      name: generateUUID(),
      type: 'audio/wav'
    }

    RNS3.put(file, options).then(response => {
      if (response.status !== 201){
        setPrediction("Failed to upload audio")
      }
    });

    console.log('Predicting bird species..')

    let response = await fetch('https://deepbirdapp.herokuapp.com/', {
       method : 'GET',
       headers: {
        'file-name': file.name
       }
      })

    let bird_prediction = await response.json()

    setPrediction(bird_prediction)
    
    console.log("DONE")
  }


  return (
    <View style={styles.container}>
      <Icon
      name='circle'
      color={recording ? '#eb3434' : '#ffffff'} />
      <Button
        title={recording ? 'Stop Recording' : 'Start Recording'}
        onPress={recording ? stopRecording : startRecording}
      />
      <Text>{"Prediction: "}</Text>
      <Text>{prediction}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center'
  },
});
