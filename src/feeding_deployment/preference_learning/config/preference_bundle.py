from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass(frozen=True)
class PreferenceDim:
    field: str
    label: str
    options: List[str]
    description: str

PREFERENCE_BUNDLE: List[PreferenceDim] = [
    PreferenceDim(
        field="microwave_time",
        label="Microwave time",
        options=["no microwave", "1 min", "2 min", "3 min"],
        description="How long food should be reheated before being served. Some users may prefer hotter food, while others prefer food closer to room temperature. Many meals begin refrigerated and are intended to be served warm or hot. Fruit and dessert meals with an intended serving temperature of cold are usually eaten without microwaving."
    ),
    PreferenceDim(
        field="occlusion_relevance",
        label="Occlusion relevance",
        options=["do not consider occlusion", "minimize left occlusion", "minimize front occlusion", "minimize right occlusion"],
        description="How important it is that the robot avoids blocking the user's view. Some users may not care about occlusion, while others may prefer the robot to avoid blocking specific directions based on their typical dining context; for example, minimizing front occlusion if they usually look at a TV, laptop, tablet, or social partner directly in front of them, minimizing left occlusion if these are to their left, or minimizing right occlusion if they are to their right."
    ),
    PreferenceDim(
        field="robot_speed",
        label="Robot speed",
        options=["slow", "medium", "fast"],
        description="The speed at which the robot moves. Some users may prefer a slower speed to feel more comfortable, or to help social partners feel comfortable, while others may prefer a faster speed to reduce overall meal time or because they are less sensitive to the robot's movements."
    ),
    PreferenceDim(
        field="skewering_axis",
        label="Skewering axis selection",
        options=["parallel to major axis", "perpendicular to major axis"],
        description="The direction in which the robot inserts the fork into food when acquiring a bite. Parallel skewering can produce narrower bites that may be easier to eat for users with limited mouth opening. Perpendicular skewering can increase bite acquisition success."
    ),
    PreferenceDim(
        field="web_interface_confirmation",
        label="Web interface confirmation",
        options=["yes", "no"],
        description="Whether the system requires explicit confirmation from the user, through the web interface, that bite acquisition has succeeded. This page allows the user to retry bite acquisition if it fails, but some users may prefer to skip this step to reduce interaction time, even if it means the robot might attempt to transfer an empty fork or a fork with a failed bite acquisition. A user might be more comfortable skipping confirmation in certain contexts (e.g., when eating alone in a personal setting) and prefer confirmation in other contexts (e.g., when eating in a social setting with a partner who might feel uncomfortable if the robot repeatedly attempts to transfer an empty fork after failed bite acquisitions). Some users may want to have confirmation in all contexts."
    ),
    PreferenceDim(
        field="transfer_mode",
        label="Outside mouth vs inside mouth transfer",
        options=["outside mouth transfer", "inside mouth transfer"],
        description="How food is delivered to the mouth: outside-mouth transfer (the robot stops just outside the mouth, and the user leans forward to take the bite) versus inside-mouth transfer (the robot inserts the food directly into the mouth). Preference may depend on the user's physical capabilities (e.g., whether they can lean forward comfortably), their comfort with the robot moving close to their mouth, or their affective state (e.g., preferring inside-mouth transfer when fatigued)."
    ),
    PreferenceDim(
        field="outside_mouth_distance",
        label="For outside-mouth transfer: distance from the mouth",
        options=["not applicable", "near", "medium", "far"],
        description="This parameter only applies when transfer_mode is outside mouth transfer. If inside mouth transfer is used, the value should be not applicable. When using outside-mouth transfer, this determines how far from the mouth the robot stops before the user takes the bite. Far refers to the farthest distance the user can comfortably reach, and near refers to very close (2-4 cm away) from the mouth. This depends on the user's comfort with the robot and their affective state (e.g., preferring a closer distance when fatigued)."
    ),
    PreferenceDim(
        field="convey_robot_ready_for_initiating_transfer",
        label="Convey robot is ready for initiating transfer",
        options=["speech", "LED", "speech + LED", "no cue"],
        description="How the robot signals to the user that it is ready to initiate transfer of a bite, sip, or mouth wiping."
    ),
    PreferenceDim(
        field="detect_user_ready_for_initiating_transfer_feeding",
        label="Detect User Ready for Initiating Transfer - FEEDING",
        options=["open mouth", "button", "autocontinue"],
        description="How the robot determines that the user is ready to initiate bite transfer. open mouth: readiness is detected from the user opening their mouth (this can be a challenge in social settings when a user tends to open their mouth for talking); button: the user explicitly presses a physical button (this can be cumbersome if the user is fatigued); autocontinue: the robot proceeds automatically after waiting for a certain timeout."
    ),
    PreferenceDim(
        field="detect_user_ready_for_initiating_transfer_drinking",
        label="Detect User Ready for Initiating Transfer - DRINKING",
        options=["open mouth", "button", "autocontinue"],
        description="How the robot determines that the user is ready to initiate sip transfer. open mouth: readiness is detected from the user opening their mouth (this can be a challenge in social settings when a user tends to open their mouth for talking); button: the user explicitly presses a physical button (this can be cumbersome if the user is fatigued); autocontinue: the robot proceeds automatically after waiting for a certain timeout."
    ),
    PreferenceDim(
        field="detect_user_ready_for_initiating_transfer_wiping",
        label="Detect User Ready for Initiating Transfer - MOUTH WIPING",
        options=["open mouth", "button", "autocontinue"],
        description="How the robot determines that the user is ready to initiate transfer of mouth wiper. open mouth: readiness is detected from the user opening their mouth (this can be a challenge in social settings when a user tends to open their mouth for talking); button: the user explicitly presses a physical button (this can be cumbersome if the user is fatigued); autocontinue: the robot proceeds automatically after waiting for a certain timeout."
    ),
    PreferenceDim(
        field="convey_robot_ready_for_completing_transfer",
        label="Convey robot is ready for completing transfer",
        options=["speech", "LED", "speech + LED", "no cue"],
        description="How the robot signals to the user that the tool has reached the transfer location and the user can complete the transfer by taking a bite, sip, or mouth wiping."
    ),
    PreferenceDim(
        field="detect_user_completed_transfer_feeding",
        label="Detect User Completed Transfer - FEEDING",
        options=["perception", "button", "autocontinue"],
        description="How the robot determines that the user has finished taking a bite. perception: the robot detects completion using a force-torque sensor on the fork (very reliable); button: the user explicitly signals completion by physically pressing a button; autocontinue: the robot proceeds automatically after a certain timeout."
    ),
    PreferenceDim(
        field="detect_user_completed_transfer_drinking",
        label="Detect User Completed Transfer - DRINKING",
        options=["perception", "button", "autocontinue"],
        description="How the robot determines that the user has finished drinking. perception: the robot detects completion using a head-nod gesture (which may feel unnatural in social settings); button: the user explicitly signals completion by physically pressing a button; autocontinue: the robot proceeds automatically after a certain timeout."
    ),
    PreferenceDim(
        field="detect_user_completed_transfer_wiping",
        label="Detect User Completed Transfer - MOUTH WIPING",
        options=["perception", "button", "autocontinue"],
        description="How the robot determines that the user has finished mouth wiping. perception: the robot detects completion using a head-nod gesture (which may feel unnatural in social settings); button: the user explicitly signals completion by physically pressing a button; autocontinue: the robot proceeds automatically after a certain timeout."
    ),
    PreferenceDim(
        field="retract_between_bites",
        label="Retract between bites",
        options=["yes", "no"],
        description="Whether the robot moves to a retract position between tasks to avoid obstructing the user's view. However, moving to this position makes the meal take longer, so some users may prefer to skip this step to reduce meal time, even if it means the robot might obstruct the user's view for a longer duration during the meal. A user might be more comfortable skipping retracting in certain contexts (e.g., when eating alone in a personal setting) and prefer retracting in other contexts (e.g., when working, watching TV, or eating in a social setting with a partner who might feel uncomfortable if the robot obstructs their view for a long duration)."
    ),
    PreferenceDim(
        field="bite_dipping_preference",
        label="Bite dipping preference",
        options=["do not dip", "less", "more"],
        description="How much sauce should be applied when dipping. This depends on the user's personal preference, as well as the context (e.g., some users might prefer more dipping when eating alone in a personal setting and less dipping when eating in a social setting to avoid messiness). Choose do not dip when the user prefers not to dip or when the meal does not have any dippable items or sauces."
    ),
    PreferenceDim(
        field="wait_before_autocontinue_seconds",
        label="Time to wait before autocontinue",
        options=["10 sec", "100 sec", "1000 sec"],
        description="How long the robot waits before automatically continuing to the next bite, sip, or mouth wiping if the user does not intervene. Some users may prefer a shorter wait time to reduce meal time, while others may prefer a longer wait time to give themselves more time to intervene if needed, especially in contexts where they might be more distracted (e.g., when eating in a social setting with a partner or when watching TV)."
    ),
]