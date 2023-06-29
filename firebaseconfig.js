const config={
  apiKey: "AIzaSyA9Dbe4nXwzZmBuWzxzrSrYRV9aXKMi1iE",
  authDomain: "interviewbot-4d76b.firebaseapp.com",
  projectId: "interviewbot-4d76b",
  storageBucket: "interviewbot-4d76b.appspot.com",
  messagingSenderId: "1007191484062",
  appId: "1:1007191484062:web:f86ee28e5c7174a7353d8f",
  measurementId: "G-Q912F5TX87"
};

export function getFirebaseConfig() {
    if (!config || !config.apiKey) {
      throw new Error('No Firebase configuration object provided.' + '\n' +
      'Add your web app\'s configuration object to firebase-config.js');
    } else {
      return config;
    }
  }
