class Constellation:
    def __init__(self, name, specific_questions, visibility):
        self.name = name
        self.specific_questions = specific_questions
        self.visibility = visibility

    def get_specific_question(self, question_index):
        if question_index < len(self.specific_questions):
            return self.specific_questions[question_index]
        return None

    def is_visible(self, country, season):
        """Check if the constellation is visible in the specified country and season."""
        return self.visibility.get(country, {}).get(season, 'None') != 'None'
    

class KnowledgeBase:
    # Common questions related to visibility
    common_questions = [
        "1.Is it visible from the United States during winter?",
        "2.Is it visible from Japan during spring?",
        "3.Is it visible from Australia during summer?",
        "4.Is it part of the zodiac constellations?",
        "5.Is it associated with mythology?",
        "6.Does it contain any notable stars?",
        "7.Is it prominently visible in the northern hemisphere during summer?",
        "8.Is it visible all year round in the northern hemisphere?",
        "9.Is it located near the Milky Way?",
        "10.Is it associated with any specific season?",
        "11.Is it visible in the southern hemisphere during winter?",
        "12.Does it have a distinctive shape or pattern?",
        "13.Is it used for navigation?",
        "14.Does it contain star clusters or nebulae?",
        "15.Is it recognized in cultural or historical references?"
    ]

    # Map common questions to constellations
    common_visibility = {
        common_questions[0]: ["Orion", "Canis Major", "Taurus", "Pleiades", "Gemini", "Ursa Major"],
        common_questions[1]: ["Leo", "Gemini", "Cygnus", "Lyra"],
        common_questions[2]: ["Sagittarius", "Aquila", "Canis Major"],
        common_questions[3]: ["Taurus", "Gemini", "Leo", "Sagittarius"],
        common_questions[4]: ["Orion", "Canis Major", "Taurus", "Sagittarius", "Leo", "Gemini"],
        common_questions[5]: ["Orion", "Canis Major", "Leo", "Gemini", "Aquila", "Ursa Major"],
        common_questions[6]: ["Cygnus", "Lyra", "Aquila", "Sagittarius"],
        common_questions[7]: ["Ursa Major", "Cassiopeia", "Ursa Minor"],
        common_questions[8]: ["Cygnus", "Aquila", "Orion"],
        common_questions[9]: ["Orion", "Taurus", "Gemini", "Leo"],
        common_questions[10]: ["Canis Major", "Sagittarius"],
        common_questions[11]: ["Orion", "Ursa Major", "Cassiopeia", "Leo", "Taurus"],
        common_questions[12]: ["Ursa Major", "Orion", "Canis Major"],
        common_questions[13]: ["Pleiades", "Orion", "Sagittarius"],
        common_questions[14]: ["Orion", "Taurus", "Gemini", "Leo"]
    }

    # Define specific questions and visibility for each constellation
    constellations = [
        Constellation(
            "Orion",
            [
                "Does it contain a prominent belt of three stars?",
                "Is it associated with a hunter in mythology?",
                "Is it visible during winter months in the northern hemisphere?",
                "Does it contain the bright star Betelgeuse?",
                "Is it often depicted as a hunter in various cultures?"
            ],
            {
                "USA": {"Winter": "Visible", "Spring": "None", "Autumn": "Visible", "Summer": "None"},
                "Japan": {"Winter": "Visible", "Spring": "None", "Autumn": "Visible", "Summer": "None"},
                "Australia": {"Summer": "None", "Winter": "Visible", "Spring": "None", "Autumn": "Visible"}
            }
        ),
        Constellation(
            "Pleiades",
            [
                "Does it appear as a small cluster of stars?",
                "Is it often referred to as the Seven Sisters?",
                "Is it located near the constellation Taurus?",
                "Is it prominent in winter and spring in the northern hemisphere?",
                "Is it often associated with various myths in different cultures?"
            ],
            {
                "USA": {"Winter": "Visible", "Spring": "Visible", "Autumn": "None", "Summer": "None"},
                "Japan": {"Winter": "Visible", "Spring": "Visible", "Autumn": "None", "Summer": "None"},
                "Australia": {"Summer": "None", "Winter": "Visible", "Spring": "Visible", "Autumn": "None"}
            }
        ),
        Constellation(
            "Taurus",
            [
                "Does it contain a bright star named Aldebaran?",
                "Is it often depicted as a bull in mythology?",
                "Is it visible during winter months in the northern hemisphere?",
                "Is it part of the zodiac constellations?",
                "Is it located near the constellation Orion?"
            ],
            {
                "USA": {"Winter": "Visible", "Spring": "Visible", "Autumn": "None", "Summer": "None"},
                "Japan": {"Winter": "Visible", "Spring": "Visible", "Autumn": "None", "Summer": "None"},
                "Australia": {"Summer": "None", "Winter": "Visible", "Spring": "Visible", "Autumn": "None"}
            }
        ),
        Constellation(
            "Canis Major",
            [
                "Does it contain the brightest star in the night sky?",
                "Is it associated with a hunting dog in mythology?",
                "Is it visible during winter months in the northern hemisphere?",
                "Is it located near the constellation Orion?",
                "Is its brightest star often used as a navigation reference?"
            ],
            {
                "USA": {"Winter": "Visible", "Spring": "None", "Autumn": "None", "Summer": "None"},
                "Japan": {"Winter": "Visible", "Spring": "None", "Autumn": "None", "Summer": "None"},
                "Australia": {"Summer": "Visible", "Winter": "None", "Spring": "None", "Autumn": "None"}
            }
        ),
        Constellation(
            "Leo",
            [
                "Is it associated with a lion in mythology?",
                "Is it a prominent zodiac constellation?",
                "Is it visible in spring in the northern hemisphere?",
                "Does it contain a bright star named Regulus?",
                "Is it located near the constellation Virgo?"
            ],
            {
                "USA": {"Spring": "Visible", "Summer": "Visible", "Autumn": "None", "Winter": "None"},
                "Japan": {"Spring": "Visible", "Summer": "Visible", "Autumn": "None", "Winter": "None"},
                "Australia": {"Autumn": "Visible", "Winter": "None", "Spring": "None", "Summer": "None"}
            }
        ),
        Constellation(
            "Gemini",
            [
                "Does it consist of two bright stars near each other, representing twins?",
                "Is it associated with Greek mythology and twin brothers?",
                "Is it visible prominently during winter in the northern hemisphere?",
                "Is it part of the zodiac constellations?",
                "Is it found near the constellation Orion?"
            ],
            {
                "USA": {"Winter": "Visible", "Spring": "Visible", "Autumn": "None", "Summer": "None"},
                "Japan": {"Winter": "Visible", "Spring": "Visible", "Autumn": "None", "Summer": "None"},
                "Australia": {"Summer": "None", "Winter": "Visible", "Spring": "Visible", "Autumn": "None"}
            }
        ),
        Constellation(
            "Bootes",
            [
                "Does it contain a very bright orange star at its center?",
                "Is it associated with a mythological herdsman?",
                "Does it appear prominently in spring in the northern hemisphere?",
                "Is it located near the constellation Virgo?",
                "Is its shape often described as kite-like?"
            ],
            {
                "USA": {"Spring": "Visible", "Summer": "Visible", "Autumn": "None", "Winter": "None"},
                "Japan": {"Spring": "Visible", "Summer": "Visible", "Autumn": "None", "Winter": "None"},
                "Australia": {"Autumn": "Visible", "Winter": "Visible", "Spring": "None", "Summer": "None"}
            }
        ),
        Constellation(
            "Lyra",
            [
                "Does it contain a very bright blue star?",
                "Is it associated with a musical instrument in mythology?",
                "Is it prominent in the northern hemisphere's summer?",
                "Does it contain a famous ring-shaped nebula?",
                "Is it part of the Summer Triangle?"
            ],
            {
                "USA": {"Summer": "Visible", "Spring": "None", "Autumn": "Visible", "Winter": "None"},
                "Japan": {"Summer": "Visible", "Spring": "None", "Autumn": "Visible", "Winter": "None"},
                "Australia": {"Winter": "Visible", "Summer": "None", "Spring": "None", "Autumn": "None"}
            }
        ),
        Constellation(
            "Canis Minor",
            [
                "Does it contain a very bright star but is a small constellation?",
                "Is it associated with a smaller dog in mythology?",
                "Is it positioned near the constellation Canis Major?",
                "Does it have two primary stars and little else?",
                "Is it most visible in the winter months?"
            ],
            {
                "USA": {"Winter": "Visible", "Spring": "None", "Autumn": "None", "Summer": "None"},
                "Japan": {"Winter": "Visible", "Spring": "None", "Autumn": "None", "Summer": "None"},
                "Australia": {"Summer": "None", "Winter": "Visible", "Spring": "None", "Autumn": "None"}
            }
        ),
        Constellation(
            "Sagittarius",
            [
                "Is it often depicted as a centaur in mythology?",
                "Is it part of the zodiac constellations?",
                "Is it visible prominently during summer in the southern hemisphere?",
                "Does it contain a bright area called the Galactic Center?",
                "Is it located near the constellation Scorpius?"
            ],
            {
                "USA": {"Summer": "Visible", "Spring": "None", "Autumn": "None", "Winter": "None"},
                "Japan": {"Summer": "Visible", "Spring": "None", "Autumn": "None", "Winter": "None"},
                "Australia": {"Winter": "None", "Summer": "Visible", "Spring": "None", "Autumn": "Visible"}
            }
        ),
        Constellation(
            "Aquila",
            [
                "Does it represent an eagle in mythology?",
                "Is it prominent in summer in the northern hemisphere?",
                "Does it contain the bright star Altair?",
                "Is it part of the Summer Triangle?",
                "Is it found near the constellation Lyra?"
            ],
            {
                "USA": {"Summer": "Visible", "Spring": "None", "Autumn": "Visible", "Winter": "None"},
                "Japan": {"Summer": "Visible", "Spring": "None", "Autumn": "None", "Winter": "None"},
                "Australia": {"Winter": "None", "Summer": "Visible", "Spring": "None", "Autumn": "None"}
            }
        ),
        Constellation(
            "Ursa Major",
            [
                "Does it contain the famous asterism known as the Big Dipper?",
                "Is it associated with a bear in mythology?",
                "Is it circumpolar and visible all year in the northern hemisphere?",
                "Is it often used for navigation?",
                "Does it lie opposite the constellation Ursa Minor?"
            ],
            {
                "USA": {"All": "Visible"},
                "Japan": {"All": "Visible"},
                "Australia": {"Summer": "None", "Winter": "Visible", "Spring": "None", "Autumn": "None"}
            }
        ),
        Constellation(
            "Cygnus",
            [
                "Does it represent a swan in mythology?",
                "Is it part of the Summer Triangle?",
                "Is it visible prominently in summer?",
                "Does it contain a bright star called Deneb?",
                "Is it located near the constellation Lyra?"
            ],
            {
                "USA": {"Summer": "Visible", "Spring": "None", "Autumn": "Visible", "Winter": "None"},
                "Japan": {"Summer": "Visible", "Spring": "None", "Autumn": "None", "Winter": "None"},
                "Australia": {"Winter": "None", "Summer": "Visible", "Spring": "None", "Autumn": "None"}
            }
        ),
        Constellation(
            "Cassiopeia",
            [
                "Does it contain a distinctive 'W' or 'M' shape?",
                "Is it circumpolar and visible all year in the northern hemisphere?",
                "Is it associated with a mythological queen?",
                "Does it lie along the Milky Way?",
                "Is it found opposite the constellation Ursa Major?"
            ],
            {
                "USA": {"All": "Visible"},
                "Japan": {"All": "Visible"},
                "Australia": {"Winter": "None", "Spring": "None", "Summer": "None", "Autumn": "None"}
            }
        )
    ]
    
class Expert:
    possible_constellations = []
    
    question_index = 0
    
    specific_questions_started = False
    constellation_index = 0
    constellation_question_index = 0
    constellation_score = 0
    final_scores = None
    
    def __init__(self):
        pass
    
    def process_answer(self, question, answer):
        for constellation in KnowledgeBase.constellations:
            if answer == "yes":
                if constellation.name in KnowledgeBase.common_visibility.get(question, []):
                    if self.possible_constellations[constellation.name] > -1:
                        self.possible_constellations[constellation.name] += 1
            elif answer == "no":
                if constellation.name in KnowledgeBase.common_visibility.get(question, []):
                    self.possible_constellations[constellation.name] = -1
            elif answer == "i don't know":
                if self.possible_constellations[constellation.name] > -1:
                    self.possible_constellations[constellation.name] += 0.5  # Half credit for unknown

    def process_constellation_specific_anwer(self, constellation_name, answer):
        if answer == "yes":
            self.possible_constellations[constellation_name] += 1
            self.constellation_score += 0.2
        elif answer == "no":
            self.constellation_score += 0  # No increment needed
        elif answer == "i don't know":
            self.constellation_score += 0.1  # Half credit for unknown
        print("Constellation score is:", self.constellation_score)    
                            
    def start(self):
        self.possible_constellations = []
        self.specific_questions_started = False
        self.final_const = None
        self.question_index = 0
        self.constellation_index = 0
        self.possible_constellations = {constellation.name: 0 for constellation in KnowledgeBase.constellations}
        
        
    def get_question(self):
        final = True;
        data = {}
        data['question'] = False
        data['posible_constelations'] = self.possible_constellations
        #print(data['posible_constelations'])
        question = None
        for question in KnowledgeBase.common_questions[self.question_index:]:
            self.question_index = self.question_index + 1
            
            q_cons = KnowledgeBase.common_visibility.get(question, [])
            to_skip = True
            for q_con in q_cons:
                if self.possible_constellations[q_con] != -1:
                    to_skip = False
                    break
                
            if to_skip:
                continue  # skip questions containing the constellations that received a no to a question
            else:
                break;
        
        if (not question is None):
            data['question'] = question       
        if (data['question'] == False or self.question_index >= len(KnowledgeBase.common_questions)):
            data['question'] = None
            if (self.specific_questions_started == False):
                self.final_scores = {}
                # Filter possible constellations based on their scores
                self.possible_constellations = {name: score for name, score in self.possible_constellations.items() if score > 0}
                self.specific_questions_started = True
                
            if not self.possible_constellations:
                data['not_found'] = True                
                return data
            
            
            data = self.get_constellation_specific_question()
            
        return data;           


            
    def get_constellation_specific_question(self):
        print(self.possible_constellations)
        data = {}
        data['question'] = None
       
         # Ask specific questions for each remaining possible constellation       
        for constellation in KnowledgeBase.constellations[self.constellation_index:]:            
            if constellation.name in self.possible_constellations:
                #print(f"get specific question for constellatio:{constellation.name} for index {self.constellation_question_index}")
                
                if (self.constellation_question_index == 0):
                    self.constellation_score = 0
                if (self.constellation_question_index < len(constellation.specific_questions)):
                    
                    data['question'] = constellation.get_specific_question(self.constellation_question_index)
                    data['constellation_name'] = constellation.name

                else:
                    self.constellation_index = self.constellation_index + 1    
                    
                if (data['question'] is None):
                    
                    self.final_scores[constellation.name] = self.constellation_score * 100  
                    self.constellation_question_index = 0
                    print("set final score for:", constellation.name, self.final_scores)
                else:
                    self.constellation_question_index = self.constellation_question_index+1
                    return data    
            else:
                self.constellation_index = self.constellation_index + 1        
    
        if (data['question'] == False or data['question'] is None or self.constellation_index >= len(KnowledgeBase.constellations)):
            # Find the constellation with the highest score
            #print("final scores=", self.final_scores)
            identified_constellation = max(self.final_scores, key=self.final_scores.get)
            score = self.final_scores[identified_constellation]
        
            data['identified_constellation'] = identified_constellation
            data['final_constelations'] = self.final_scores
            data['probability'] = round(score, 2)
            data['final'] = True
                
        return data
    
# Funcția principală pentru a rula sistemul expert
if __name__ == "__main__":
    expert = Expert()
    expert.start()
    answers_pleiades = ['yes', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'no', 'no', 'no', 'yes']
    
    answers = ['yes', 'no', 'no', 'yes', "i don't know", 'no', 'yes', 'no', 'no', 'no'] #{'Bootes': 0.5, 'Canis Minor': 0.5}
    answers = ['no', "i don't know", 'yes', 'no', "i don't know", 'yes', 'no']
    answers = ['no', 'no', 'no', 'no', 'no', 'no', 'no', 'no']
    answers = []
    for answer in  answers:
        data = expert.get_question()
        print(f"{data['question']} (yes/no/i don't know) ")
        print(answer)
        expert.process_answer(data['question'], answer)
        
    while (True):            
        data = expert.get_question()
        print(data)
        if (('question' not in data) or data['question'] == False or data['question'] is None):
            #print(data)
            if ('not_found' in data and data['not_found']):
                print('Constelations not found!')
            else:
                print(f"\nThe constellation you are observing is likely: {data['identified_constellation']} (Probability: {data['probability']:.2f}%)")
                print("\nThe possible solutions were:", data['final_constelations'] )
            break
        else:
            answer = input(f"{data['question']} (yes/no/i don't know) ").strip().lower()
            if 'constellation_name' in data and data['constellation_name']:
                expert.process_constellation_specific_anwer(data['constellation_name'], answer)
            else:
                expert.process_answer(data['question'], answer)
        
    
    