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
  const https = require('https')

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
      setPrediction("");
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
      type: 'audio/mp3'
    }
    RNS3.put(file, options).then(response => {
      if (response.status !== 201){
        setPrediction("Failed to upload audio")
      }
      // console.log(response.body);
    });
    console.log('Predicting bird species..')
    const predictOptions = {
      hostname: 'replacewithherokuurl.com',
      port: 5000,
      path: '/predict',
      method: 'GET'
    }

    const req = https.request(predictOptions, res => {
      console.log(`statusCode: ${res.statusCode}`)

      res.on('data', d => {
        setPrediction(d)
      })
    })

    req.on('error', error => {
      console.error(error)
    })

    req.end()
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
      <Text>{"Prediction: \n"}</Text>
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
