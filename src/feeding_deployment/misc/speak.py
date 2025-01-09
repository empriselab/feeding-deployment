import rospy
from std_msgs.msg import String
import tempfile
from gtts import gTTS
from playsound import playsound

class Speak:
    def __init__(self):
        rospy.init_node('Speak', anonymous=True)
        rospy.Subscriber("/speak", String, self.callback, queue_size=10)
        print("Speak node initialized")

    # Speak the text
    def callback(self, msg):        
        text = msg.data
        print("Speaking: ", text)

        # Convert text to speech and play
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as voice:
            gTTS(text=text, lang="en").write_to_fp(voice)
            voice.flush()  # Ensure data is written to file
            playsound(voice.name)

if __name__ == "__main__":
    try:
        Speak()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass