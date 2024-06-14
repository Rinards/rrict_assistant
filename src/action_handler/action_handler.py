def handle_input(user_input):
    if user_input == None:
        answer = "Enter something!"
        print(answer)
    else:
        handle_action()
def handle_action():
    if label == "ALARM":
        print("opening alarm app")
    elif label == "CANCEL":
        print("cancelling request")
    elif label == "DATE":
        print("opening calendar app")
    elif label == "REPEAT":
        print("repeating request")
    elif label == "TIMER":
        print("opening timer app")
    elif label == "TRANSLATE":
        print("Starting translation")
    elif label == "WEATHER":
        print("opening weather app")
    elif label == "WHAT_CAN_I_ASK_YOU":
        print("I can help you with the following: \n"
              "1. Set an alarm \n"
              "2. doing calculations \n"
              "3. Check the date \n"
              "4. Repeat the last action \n"
              "5. Set a timer \n"
              "6. Translate a word \n"
              "7. Check the weather \n"
              "8. Talk to you \n"
              "9. Do web search \n"
              "10. Open calendar \n"
              "11. Do system actions \n"
              "12. Cancel requests \n")
    elif label == "CALCULATOR":
        print("opening calculator app")
    elif label == "WEB_SEARCH":
        print("doing web search")
    elif label == "YES":
        print("confirming")
    elif label == "NO":
        print("denying") 
    elif label == "CALENDAR":
        print("opening calendar app")
    elif label == "SYSTEM_ACTION":
        print("doing system actions")
    elif label == "TALK_TO_ME":
        print("talking to you")            
user_input = input("Please enter something: ")
label = None
handle_input(user_input)



                           