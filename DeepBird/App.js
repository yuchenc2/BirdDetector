import { StatusBar } from 'expo-status-bar';
import React, { Component } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { Button } from 'react-native-elements';
import Icon from 'react-native-vector-icons/FontAwesome';

export default class App extends Component {
  state = {
    recordingState: 'Press to Record'
  }

  onPressRecord = () => {
    this.setState({
      recordingState: this.state.recordingState == 'Recording...' 
        ? 'Press to Record' : 'Recording...'
    })
  }

  render() {
    return (
      <View style={styles.container}>
        <Text> {this.state.recordingState} </Text>
        <Button 
          icon={
            <Icon
              name="circle"
              size={20}
              color="red"
            />
          }
          title=""
          onPress={this.onPressRecord}
        />
        <StatusBar style="auto" />
      </View>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center'
  },
});
