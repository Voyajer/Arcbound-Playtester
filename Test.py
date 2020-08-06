import random
import tensorflow as tf
print("hello")

string = tf.Variable("this is a string", tf.string)
number = tf.Variable("this is a string", tf.int32)

class Keywords(object):     #this is going to have a million keywords and text effects eventually.
    def __init__(self, ):

class Mana(object):
    def __init__(self, coloridentity, wmana, umana, bmana, rmana, gmana, cmana, genericmana, azoriusmana, rakdosmana, selesnyamana, izzetmana, golgarimana):        #needs phyrexian mana and shadowmoor mana

class Card(object):
    def __init__(self, name, Mana, supertype, type, subtype, power, toughness, Keywords):



class Deck(object):
    def __init__(self):
        self.cards = []
        self.build()

    def build(self):
        Decklistfile  = open("filename", "r")
        for line in Decklistfile:
            desklist =Decklistfile.read() #i think this is right?
        file.close() #when done




class Player(object):
    def __init__(self):