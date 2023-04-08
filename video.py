from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.azure import AzureService

LARGE_FONT = 64
MEDIUM_FONT = 32

class Video(VoiceoverScene):
    def create_textbox(self, color, text):
        result = VGroup() # create a VGroup
        box = Rectangle(  # create a box
            height=text.height+0.5, width=text.width+0.75, fill_color=color, 
            fill_opacity=0.5, stroke_color=color
        )
        text.move_to(box.get_center()) # create text
        result.add(box, text) # add both objects to the VGroup
        return result


    def create_textcircle(self, color, text):
        result = VGroup() # create a VGroup
        circle = Ellipse(  # create a box
            height=text.height+0.75, width=text.width+1, fill_color=color, 
            fill_opacity=0.5, stroke_color=color
        )
        text.move_to(circle.get_center()) # create text
        result.add(circle, text) # add both objects to the VGroup
        return result


    def introduction(self, skip=False):
        '''
        Narration:
        Despite ChatGPT's popularity, not many people understand how the model was trained. 
        This video is an introduction to the techniques involved in training ChatGPT.

        Note that this is just an introduction because many of the topics are so advanced 
        that they go beyond the scope of this video.
        '''
        section_1 = "Despite ChatGPT's popularity, not many people understand how the model was trained. This video is an introduction to the techniques involved in training ChatGPT. Note that this is just an introduction because many of the topics are so advanced that they go beyond the scope of this video."
        self.next_section(skip_animations=skip)
        intro = Text("How Was ChatGPT Trained?", font_size=LARGE_FONT)

        with self.voiceover(text=section_1):
            self.play(FadeIn(intro))

        self.play(FadeOut(intro))


    def chatgpt_from_gpt3_5(self, skip=False):
        ''''
        Narration:
        To begin, the ChatGPT model was not trained from scratch. It was actually fine-tuned from
        a model called GPT-3.5. 
        
        ** NOT INCLUDED That is, the GPT-3.5 model was used as a starting point and further
        refined to fit the different needs of the ChatGPT model. **
        '''
        section_1 = "To begin, the ChatGPT model was not trained from scratch. It was actually fine-tuned from a model called GPT-3.5."
        self.next_section(skip_animations=skip)
        chatgpt = Text("ChatGPT", font_size=LARGE_FONT)
        gpt3_5 = Text("GPT-3.5", font_size=LARGE_FONT)
        arrow_1 = Arrow(start=UP*0.75, end=DOWN*0.75)

        with self.voiceover(text=section_1):
            self.play(FadeIn(chatgpt))
            self.play(chatgpt.animate.shift(DOWN*1.25))
            self.play(FadeIn(gpt3_5))
            self.play(gpt3_5.animate.shift(UP*1.25))
            self.play(FadeIn(arrow_1))

        self.play(FadeOut(gpt3_5), FadeOut(arrow_1), FadeOut(chatgpt))


    def gpt3_5_training_techniques(self, skip=False):
        '''
        Narration:
        GPT-3.5 was trained using two different methods, next token prediction 
        and masked language modeling. These two techniques combined make the model 
        capable of predicting the next word given an input sentence.
        '''
        section_1 = "GPT-3.5 was trained using two different methods, next token prediction and masked language modeling."
        section_2 = "These two techniques combined make the model capable of predicting the next word given an input sentence."
        self.next_section(skip_animations=skip)
        training_techniques = Text("Training Techniques", font_size=MEDIUM_FONT)
        ntp = Text("Next Token Prediction", font_size=MEDIUM_FONT, color=BLUE)
        mlm = Text("Masked Language Modeling", font_size=MEDIUM_FONT+2, color=RED)

        training_techniques.shift(UP*4.3)
        group = VGroup(ntp, mlm)
        group.arrange(buff=0.75)
        mlm.shift(DOWN*.05)

        with self.voiceover(text=section_1):
            self.play(training_techniques.animate.move_to(UP*3.5))
            self.wait(3)
            self.play(FadeIn(ntp))
            self.play(FadeIn(mlm))

        with self.voiceover(text=section_2):
            self.wait(4)
            self.play(FadeOut(training_techniques, shift=UP))
            self.play(FadeOut(mlm))

        self.play(ntp.animate.move_to(ORIGIN + (UP*3.5)))
        self.remove(ntp)


    def next_token_prediction(self, skip=False):
        '''
        Narration:
        Next token prediction is a training technique in which you take a partial sentence
        and use the model to try and predict the proper next word. 
        
        For instance, you could take the partial sentence the old man was blank. 

        The model would then output a list of all the words it knows and their associated 
        probabilities of being the one to fill the blank. Since happy has the highest 
        associated probability, that would be the word the model selects.

        ** NOT INCLUDED With this training technique the model would try to predict the next word, so it 
        would output a list of all the words it knows and  the probabilities that each 
        word is the right one. This model's highest predicted values were happy, tired, 
        and short. Since happy has the highest associated probability, that would be the 
        word the model selects. **
        '''
        section_1 = "Next token prediction is a training technique in which you take a partial sentence and use the model to try and predict the proper next word."
        section_2 = "For instance, you could take the partial sentence the old man was blank."
        section_3 = "The model would then output a list of all the words it knows and their associated probabilities of being the one to fill the blank. Since happy has the highest associated probability, that would be the word the model selects."
        self.next_section(skip_animations=skip)
        ntp = Text("Next Token Prediction", font_size=MEDIUM_FONT, color=BLUE)
        example_text = Text("The old man was ____", font_size=LARGE_FONT-10)
        happy = Text("happy - 23.01%", font_size=MEDIUM_FONT, t2c={'[:5]': 'GREEN', '[5:]': 'WHITE'})
        tired = Text("tired - 8.24%", font_size=MEDIUM_FONT)
        short = Text("short - 4.84%", font_size=MEDIUM_FONT)
        dot_1 = Text(".", font_size=MEDIUM_FONT)
        dot_2 = Text(".", font_size=MEDIUM_FONT)
        dot_3 = Text(".", font_size=MEDIUM_FONT)

        guess_group = VGroup(happy, tired, short, dot_1, dot_2, dot_3)
        main_group = VGroup(example_text, guess_group)

        guess_group.arrange(DOWN, buff=0.75, center=False)
        tired.align_to(guess_group, LEFT)
        short.align_to(guess_group, LEFT)
        main_group.arrange(buff=1.75)

        items = [happy, tired, short, dot_1, dot_2, dot_3]

        enter_animations = [FadeIn(x, shift=UP) for x in items]
        fade_animations = [FadeOut(x, shift=DOWN) for x in items]

        self.add(ntp.move_to(UP*3.5))

        with self.voiceover(text=section_1):
            self.wait(1)

        with self.voiceover(text=section_2):
            self.wait(1)
            self.play(Write(example_text))

        with self.voiceover(text=section_3):
            self.wait(4)
            self.play(AnimationGroup(*enter_animations, lag_ratio=0.1))

        #Clean up scene
        self.play(FadeOut(example_text), FadeOut(ntp, shift=UP))
        self.play(AnimationGroup(*fade_animations, lag_ratio=0.1))


    def masked_language_modeling(self, skip=False):
        '''
        Narration:
        In masked language modeling, you take a full sentence and replace one of the words with a special
        mask token. The model then has to predict which word correctly fills in the masked word.

        For example, the full sentence could be the old woman was tired. You could then replace the 
        word woman with a mask token and it would be the model's job to fix the sentence.

        Similarly to the previous technique, the model then predicts words that could logically fill in the mask.
        Because the word woman has the highest associated probability, that would be the word chosen.
        '''
        section_1 = "In masked language modeling, you take a full sentence and replace one of the words with a special mask token. The model then has to predict which word correctly fills in the masked word."
        section_2 = "For example, the full sentence could be the old woman was tired."
        section_3 = "You could then replace the word woman with a mask token and it would be the model's job to fix the sentence."
        section_4 = "Similarly to the previous technique, the model then predicts words that could logically fill in the mask. Because the word woman has the highest associated probability, that would be the word chosen."
        self.next_section(skip_animations=skip)
        mlm = Text("Masked Language Modeling", font_size=MEDIUM_FONT+2, color=RED)
        start = Text("The old", font_size=LARGE_FONT)
        middle = Text("woman", font_size=LARGE_FONT)
        end = Text("was tired.", font_size=LARGE_FONT)
        mask = Text("[MASK]", font_size=LARGE_FONT-3)
        woman = Text("woman - 15.89%", font_size=MEDIUM_FONT, t2c={'[:5]': 'GREEN', '[5:]': 'WHITE'})
        man = Text("man - 13.21%", font_size=MEDIUM_FONT)
        dog = Text("dog - 7.33%", font_size=MEDIUM_FONT)
        dot_1 = Text(".", font_size=MEDIUM_FONT)
        dot_2 = Text(".", font_size=MEDIUM_FONT)
        dot_3 = Text(".", font_size=MEDIUM_FONT)

        sentence_group = VGroup(start, middle, end)
        guess_group = VGroup(woman, man, dog, dot_1, dot_2, dot_3)
        main_group = VGroup()

        sentence_group.arrange(center=True, buff=0.4)
        guess_group.arrange(DOWN, buff=0.6, center=False)
        middle.shift(DOWN*0.13)
        mlm.move_to(UP*4.3)
        mask.align_to(middle, LEFT)
        man.align_to(guess_group, LEFT)
        dog.align_to(guess_group, LEFT)

        items = [woman, man, dog, dot_1, dot_2, dot_3]

        animations = [
            FadeIn(start, shift=UP),
            FadeIn(middle, shift=UP),
            FadeIn(end, shift=UP)
        ]
        enter_animations = [FadeIn(x, shift=UP) for x in items]
        fade_animations = [FadeOut(x, shift=DOWN) for x in items]

        with self.voiceover(text=section_1):
            self.play(FadeIn(mlm), mlm.animate.move_to(UP*3.5))
            
        with self.voiceover(text=section_2):
            self.wait(2)
            self.play(AnimationGroup(*animations, lag_ratio=0.1))

        with self.voiceover(text=section_3):
            self.play(FadeOut(middle, shift=DOWN))
            sentence_group.remove(middle)
            self.play(FadeIn(mask, shift=UP))
            sentence_group.add(mask)

        with self.voiceover(text=section_4):
            self.play(sentence_group.animate.scale(0.5).shift(LEFT*2.48))
            main_group.add(sentence_group, guess_group)
            main_group.arrange(buff=1.75)
            self.wait(2)
            self.play(AnimationGroup(*enter_animations, lag_ratio=0.1))

        self.play(FadeOut(mlm, shift=UP), FadeOut(start, mask, end))
        self.play(AnimationGroup(*fade_animations, lag_ratio=0.1))

    
    def gpt_3_5_capabiilities(self, skip=False):
        '''
        Narration:
        The important thing to remember about these training techniques is they create 
        a statistical representation of the language being trained on.

        This means GPT-3.5 is highly capable when tasked with
        things like text continuation, text summarization, programming, factual answering, 
        and human emulation.

        One inherent flaw with this style of training, though, is GPT-3.5 isn't good at following directions.
        '''
        section_1 = "The important thing to remember about these training techniques is they create a statistical representation of the language being trained on."
        section_2 = "This means GPT-3.5 is highly capable when tasked with things like text continuation, text summarization, programming, factual answering, and human emulation."
        section_3 = "One inherent flaw with this style of training, though, is GPT-3.5 isn't good at following directions."
        self.next_section(skip_animations=skip)
        header = Text("This creates a statistical representation of language capable of:", font_size=MEDIUM_FONT, t2c={'[:15]': 'WHITE', '[15:41]': 'GREEN', '[41:]': 'WHITE'})
        opt1 = Text("Text Continuation", font_size=MEDIUM_FONT, color=GREEN)
        opt2 = Text("Text Summarization", font_size=MEDIUM_FONT, color=GREEN)
        opt3 = Text("Programming", font_size=MEDIUM_FONT, color=GREEN)
        opt4 = Text("Factual Answering", font_size=MEDIUM_FONT, color=GREEN)
        opt5 = Text("Human Emulation", font_size=MEDIUM_FONT, color=GREEN)
        opt6 = Text("Direction Following", font_size=MEDIUM_FONT, color=RED)

        opt_group1 = VGroup(opt1, opt2, opt3).arrange(DOWN, buff=1, aligned_edge=LEFT)
        opt_group2 = VGroup(opt4, opt5, opt6).arrange(DOWN, buff=1, aligned_edge=LEFT)
        opt_group_main = VGroup(opt_group1, opt_group2).arrange(buff=3)

        header.shift(UP*3.5)

        options = [opt1, opt2, opt3, opt4, opt5]
        enter_animations = [FadeIn(x, shift=UP) for x in options]
        exit_animations = [FadeOut(x, shift=DOWN) for x in options]

        with self.voiceover(text=section_1):
            self.play(FadeIn(header))

        with self.voiceover(text=section_2):
            self.wait(3.9)
            self.play(AnimationGroup(*enter_animations, lag_ratio=1.3))

        with self.voiceover(text=section_3):
            self.wait(5.5)
            self.play(FadeIn(opt6, shift=UP))

        self.play(FadeOut(header))
        self.play(AnimationGroup(*exit_animations, lag_ratio=0.1))
        self.play(opt6.animate.move_to(ORIGIN + (UP*3.5)).set_fill(WHITE))
        self.remove(opt6)

    
    def direction_following(self, skip=False):
        '''
        Narration:
        Because of this, you could prompt GPT-3.5 to do something -
        like generate a poem about horses - 
        
        and it could simply continue your prompt by adding,
        "with a funny twist," to the end of it, instead of generating you a poem about horses.

        OpenAI saw this as a problem because in order to turn the model into a product, it
        had to be able to follow user instructions.
        '''
        section_1 = "Because of this, you could prompt GPT-3.5 to do something - like generate a poem about horses,"
        section_2 = "and it could simply continue your prompt by adding, \"with a funny twist,\" to the end of it, instead of generating you a poem about horses."
        section_3 = "OpenAI saw this as a problem because in order to turn the model into a product, it had to be able to follow user instructions."
        self.next_section(skip_animations=skip)
        direction_following = Text("Direction Following", font_size=MEDIUM_FONT)
        start = Text("Generate a poem about horses", font_size=MEDIUM_FONT)
        ending = ["with ", "a ", "funny ", "twist."]
        end = [Text(x, font_size=MEDIUM_FONT, color=GREEN) for x in ending]

        sentence_group = VGroup(start, *end).arrange(RIGHT, buff=0.15, aligned_edge=DOWN)

        enter_animations = [FadeIn(x, shift=UP) for x in end]
        exit_animations = [FadeOut(x, shift=DOWN) for x in end]

        direction_following.move_to(UP*3.5)
        end[0].shift(UP*0.1)
        end[1].shift(UP*0.1)
        end[2].shift(UP*0.02)
        end[3].shift(UP*0.12)
        
        self.add(direction_following)

        with self.voiceover(text=section_1):
            self.play(FadeIn(start))

        with self.voiceover(text=section_2):
            self.wait(2)
            self.play(AnimationGroup(*enter_animations, lag_ratio=0.1))

        with self.voiceover(text=section_3):
            self.wait(1)

        self.play(FadeOut(direction_following))
        self.play(FadeOut(start, *end))


    def RLHF(self, skip=False):
        '''
        Narration:
        The fine tuning technique used to teach GPT-3.5 how to follow directions 
        is a three step process called reinforcement learning from human feedback, or RLHF.
        
        In order, these steps are supervised fine tuning, mimic human preferences, and proximal policy optimization.
        '''
        section_1 = "The fine tuning technique used to teach GPT-3.5 how to follow directions is a three step process called reinforcement learning from human feedback, or RLHF."
        section_2 = "In order, these steps are supervised fine tuning, mimic human preferences, and proximal policy optimization."
        rlhf = Text("Reinforcement Learning from Human Feedback", font_size=MEDIUM_FONT)
        sft = Text("Supervised Fine Tuning", font_size=MEDIUM_FONT, color=RED)
        mhp = Text("Mimic Human Preferences", font_size=MEDIUM_FONT, color=GREEN)
        ppo = Text("Proximal Policy Optimization", font_size=MEDIUM_FONT, color=BLUE)

        step_group = VGroup(sft, mhp, ppo).arrange(DOWN, buff=0.75, aligned_edge=LEFT)

        rlhf.shift(UP*3.5)

        items = [sft, mhp, ppo]
        enter_animations = [FadeIn(x, shift=UP) for x in items]
        exit_animations = [FadeOut(x, shift=DOWN) for x in items[1:]]

        with self.voiceover(text=section_1):
            self.play(FadeIn(rlhf))

        with self.voiceover(text=section_2):
            self.wait(1)
            self.play(AnimationGroup(*enter_animations, lag_ratio=0.1))

        self.play(FadeOut(rlhf))
        self.play(AnimationGroup(*exit_animations, lag_ratio=0.1))
        self.play(sft.animate.move_to(ORIGIN + (UP*3.5)))
        self.remove(sft)


    def SFT(self, skip=False):
        '''
        Narration:
        In the supervised fine tuning step, OpenAI hired people to come up with and answer
        twelve to fifteen thousand high-quality direction-following prompts.

        These prompt-answer pairs were then passed to the GPT-3.5 model and it was trained using the techniques 
        previously mentioned.
        '''
        section_1 = "In the supervised fine tuning step, OpenAI hired people to come up with and answer twelve to fifteen thousand high-quality direction-following prompts."
        section_2 = "These prompt-answer pairs were then passed to the GPT-3.5 model and it was trained using the techniques previously mentioned."
        self.next_section(skip_animations=skip)
        sft = Text("Supervised Fine Tuning", font_size=MEDIUM_FONT, color=RED)
        people = [SVGMobject("./images/person.svg").scale(0.5) for _ in range(3)]
        model = SVGMobject("./images/model.svg").scale(0.6)
        prompts = [self.create_textbox(BLUE, Text("Prompt", font_size=MEDIUM_FONT)) for _ in range(3)]
        answers = [self.create_textcircle(GREEN, Text("Answer", font_size=MEDIUM_FONT)) for _ in range(3)]

        circle_width = answers[0].width

        person_group = VGroup(*people).arrange(DOWN, buff=0.75)
        prompt_goup = VGroup(*prompts).arrange(DOWN, buff=0.85)
        answer_group = VGroup(*answers).arrange(DOWN, buff=0.68)

        person_group.shift(LEFT*2)
        model.shift(RIGHT*2)
        prompt_goup.shift(LEFT*4)
        answer_group.shift(LEFT*2)
        answer_group.stretch_to_fit_width(0.01)

        sft.shift(UP*3.5)
        self.add(sft)

        with self.voiceover(text=section_1):
            self.play(FadeIn(person_group))
            self.play(FadeIn(model))
            self.wait(1)
            self.play(FadeIn(prompt_goup))
            self.play(prompt_goup.animate.move_to(person_group))
            self.play(prompt_goup.animate.stretch(0, 0))
            self.remove(prompt_goup)
            self.play(FadeIn(answer_group))
            self.play(answer_group.animate.stretch_to_fit_width(circle_width))
        
        with self.voiceover(text=section_2):
            self.wait(2)
            self.play(AnimationGroup(*[x.animate.move_to(model) for x in answers], lag_ratio=0))

        self.play(FadeOut(sft, shift=UP))
        self.play(FadeOut(person_group, answer_group, model))
        self.wait(1)

    
    def MHP(self, skip=False):
        '''
        Narration:
        In the mimic human preferences step, a second model is trained to rank the outputs of the model
        from most helpful to least helpful. In order to do this, OpenAI made a list of about 40 thousand
        prompts and had the model fine-tuned in the previous step generate between 4 and 9 different
        responses to each prompt. 
        
        Then, they employed a large number of people to rank the generated responses from best to worst.

        These prompts, along with their rankings, were then passed into the second model for it to learn 
        what types of responses humans prefer.
        '''
        section_1 = "In the mimic human preferences step, a second model is trained to rank the outputs of the model from most helpful to least helpful. In order to do this, OpenAI made a list of about 40 thousand prompts and had the model fine-tuned in the previous step generate between 4 and 9 different responses to each prompt. "
        section_2 = "Then, they employed a large number of people to rank the generated responses from best to worst."
        section_3 = "These prompts, along with their rankings, were then passed into the second model for it to learn what types of responses humans prefer."
        self.next_section(skip_animations=skip)
        mhp = Text("Mimic Human Preferences", font_size=MEDIUM_FONT, color=GREEN)
        answers = [self.create_textcircle(GREEN, Text(f"Answer {x+1}", font_size=MEDIUM_FONT)) for x in range(4)]
        person = SVGMobject("./images/person.svg").scale(0.5)
        model_1 = SVGMobject("./images/model.svg").scale(0.6)
        SFT_model = Text("SFT Model", font_size=MEDIUM_FONT)
        model_2 = SVGMobject("./images/model.svg").scale(0.6)
        MHP_Model = Text("MHP Model", font_size=MEDIUM_FONT)

        answer_group = VGroup(*answers)

        mhp.move_to(UP*4.3)
        model_1.shift(LEFT*5)
        SFT_model.shift(LEFT*5+DOWN)
        model_2.shift(RIGHT*5)
        MHP_Model.shift(RIGHT*5+DOWN)
        
        for answer in answers:
            answer.move_to(model_1)

        with self.voiceover(text=section_1):
            self.play(FadeIn(mhp), mhp.animate.move_to(UP*3.5))
            self.play(FadeIn(model_1, SFT_model, person, model_2, MHP_Model))
            self.wait(8)
            self.play(answer_group.animate.arrange(DOWN, buff=0.4).move_to(LEFT*2.5))

        with self.voiceover(text=section_2):
            self.wait(1)
            self.play(AnimationGroup(*[x.animate.move_to(person) for x in answers], lag_ratio=0))
            self.wait(2)
            answer_group.shuffle_submobjects()
            self.play(answer_group.animate.arrange(DOWN, buff=0.4).move_to(RIGHT*2.5))

        with self.voiceover(text=section_3):
            self.wait(2)
            self.play(AnimationGroup(*[x.animate.move_to(model_2) for x in answers], lag_ratio=0))

        self.play(FadeOut(mhp, shift=UP))
        self.play(FadeOut(model_1, SFT_model, person, model_2, MHP_Model, answer_group))


    def PPO(self, skip=False):
        '''
        Narration:
        In proximal policy optimization, the two models previously trained are used in conjunction to create ChatGPT.
        
        First, the model trained during supervised fine-tuning is tasked with generating a response 
        from a random prompt in the set.

        Next, the model trained in the mimic human preferences step, rates how helpful the response is.

        Finally, the model trained during supervised fine-tuning uses the rating given by the MHP model 
        and is either rewarded for a good response or punished for a bad response using reinforcement learning.
        '''
        section_1 = "In proximal policy optimization, the two models previously trained are used in conjunction to create ChatGPT."
        section_2 = "First, the model trained during supervised fine-tuning is tasked with generating a response from a random prompt in the set."
        section_3 = "Next, the model trained in the mimic human preferences step, rates how helpful the response is."
        section_4 = "Finally, the model trained during supervised fine-tuning uses the rating given by the MHP model and is either rewarded for a good response or punished for a bad response using reinforcement learning."
        self.next_section(skip_animations=skip)
        ppo = Text("Proximal Policy Optimization", font_size=MEDIUM_FONT, color=BLUE)
        model_1 = SVGMobject("./images/model.svg").scale(0.6)
        SFT_model = Text("SFT Model", font_size=MEDIUM_FONT)
        model_2 = SVGMobject("./images/model.svg").scale(0.6)
        MHP_Model = Text("MHP Model", font_size=MEDIUM_FONT)
        answer = self.create_textcircle(GREEN, Text("Answer", font_size=MEDIUM_FONT))
        rating = self.create_textbox(BLUE, Text("Rating", font_size=MEDIUM_FONT))
        good = self.create_textcircle(GREEN, Text("👍", font_size=MEDIUM_FONT))
        bad = self.create_textcircle(RED, Text("👎", font_size=MEDIUM_FONT))

        ppo.move_to(UP*4.3)
        model_1.shift(LEFT*4)
        SFT_model.shift(LEFT*4+DOWN)
        model_2.shift(RIGHT*4)
        MHP_Model.shift(RIGHT*4+DOWN)
        answer.move_to(model_1)
        rating.move_to(model_2)
        good.move_to(model_1)
        good.shift(UP*1.3)
        bad.move_to(model_1)
        bad.shift(DOWN*2)

        with self.voiceover(text=section_1):
            self.play(FadeIn(ppo), ppo.animate.move_to(UP*3.5))
            self.play(FadeIn(model_2, MHP_Model, model_1, SFT_model))

        with self.voiceover(text=section_2):
            self.play(FadeIn(answer))

        with self.voiceover(text=section_3):
            self.play(answer.animate.move_to(model_2))
            self.wait(3)
            self.play(FadeOut(answer), FadeIn(rating))

        with self.voiceover(text=section_4):
            self.play(rating.animate.move_to(model_1))
            self.wait(3)
            self.play(FadeOut(rating))
            self.wait(0.75)
            self.play(FadeIn(good, shift=UP))
            self.wait(0.85)
            self.play(FadeOut(good), FadeIn(bad, shift=DOWN))
            self.wait(1.5)
            self.play(FadeOut(bad))

        self.play(FadeOut(ppo, shift=UP))
        self.play(FadeOut(model_1, SFT_model, model_2, MHP_Model))


    def outro(self, skip=False):
        '''
        Narration:
        For some closing thoughts. The first step of reinforcement learning from human 
        feedback is the hardest and most expensive because you need to select and pay 
        an unbiased group to come up with prompts and answers. Because of this, this step is only done once. 
        
        The other two steps can be repeated as many times as necessary to steer the 
        model in the most helpful direction possible. 
        
        Thank you for watching.
        '''
        section_1 = "For some closing thoughts. The first step of reinforcement learning from human feedback is the hardest and most expensive because you need to select, and pay, an unbiased group to come up with prompts and answers. Because of this, this step is only done once."
        section_2 = "The other two steps can be repeated as many times as necessary to steer the model in the most helpful direction possible."
        section_3 = "Thank you for watching."
        self.next_section(skip_animations=skip)
        closing_thoughts = Text("Closing Thoughts", font_size=MEDIUM_FONT)
        sft = Text("Supervised Fine Tuning", font_size=MEDIUM_FONT, color=RED)
        mhp = Text("Mimic Human Preferences", font_size=MEDIUM_FONT, color=GREEN)
        ppo = Text("Proximal Policy Optimization", font_size=MEDIUM_FONT, color=GREEN)
        arrow = CurvedDoubleArrow(LEFT*3.1, LEFT*3.1+DOWN)

        step_group = VGroup(sft, mhp, ppo).arrange(DOWN, buff=0.75, aligned_edge=LEFT)

        closing_thoughts.move_to(UP*4.3)

        with self.voiceover(text=section_1):
            self.play(FadeIn(closing_thoughts), closing_thoughts.animate.move_to(UP*3.5))
            self.wait(3)
            self.play(AnimationGroup(*[FadeIn(x, shift=UP) for x in step_group], lag_ratio=0.1))

        with self.voiceover(text=section_2):
            self.play(FadeIn(arrow))

        self.wait(0.5)
        self.play(FadeOut(step_group, arrow))

        with self.voiceover(text=section_3):
            self.play(FadeOut(closing_thoughts, shift=UP))




    # Function to animate the video
    def construct(self):
        self.set_speech_service(AzureService(voice="en-US-GuyNeural", style="newscast")) # en-US-AriaNeural newscast-casual

        self.introduction()
    
        self.chatgpt_from_gpt3_5()
        
        self.gpt3_5_training_techniques()

        self.next_token_prediction()

        self.masked_language_modeling()

        self.gpt_3_5_capabiilities()

        self.direction_following()

        self.RLHF()

        self.SFT()

        self.MHP()

        self.PPO()

        self.outro()