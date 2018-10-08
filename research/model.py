import random
from time import gmtime, strftime
import autograd.numpy as np
import csv
import ast
import os
import json


#############################################################################
#
#
#                GET RID OF THE GLOBAL VARIABLE ALLPEOPLE
#
#
#############################################################################

def defaultAffFunc(person_obj):
    try:
        return len(person_obj.diagnoses) and person_obj.diagnoses[0][0] == person_obj.pedigree.probandDisease[0][0]
    except:
        print('error calling the affected property of Person object')


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

allPeople = {}

def idToPerson(personId,pedigree):
    global allPeople
    key = str(pedigree.studyID)+str(personId)
    if(key in allPeople):
        return allPeople[key]
    elif(int(personId) < 0):
        # add the person for the moment, but don't update any attributes
        jsonToUse = {'shapeName':'diamond',
            'Id':personId,
            'X':-1,
            'Y':-1,
            'parents':'',
            'adoptiveParents':'',
            'mateKids':'[]',
            'diagnosis':'',
            'ageAtVisit':'',
            'otherInfo':'',
            'numbPersons':'',
            'divorcedList':'',
            'consanguinityList':'',
            'zygoticList':'',
            'zygoticType':'',
            'noChildrenList':'',
            'infertileList':'',
            'dead':'',
            'ageOfDeath':'',
            'stillBirth':'',
            'ageOfStillBirth':'',
            'prematureDeath':'',
            'typeOfPrematureDeath':'',
            'unknownFamilyHistory':'',
            'proband':'',
            'consultand':'',
            'carrier':'',
            'pregnant':'',
            'surrogate':'',
            'donor':''}
        addPerson(pedigree,jsonToUse)
        return allPeople[key]
    else:
        assert 0,'The Id: '+str(personId)+' for '+str(pedigree.studyID)+' is invalid!'

def addPerson(pedigree,currentJSON):
    global allPeople
    key = str(pedigree.studyID)+str(currentJSON['Id'])
    adding = Person(currentJSON,pedigree)
    allPeople[key] = adding
    pedigree.family.append(adding)
    return adding

class Person:

    def __init__(self,jsonObject,pedigree):

        self.jsonObject = jsonObject
        self.studyID = pedigree.studyID
        self.pedigree = pedigree
        self.familyNumbers = []

        self.setAffectedFunctions(defaultAffFunc)

    def myAssert(self,invalid=None,cantFind=None):
        if(cantFind):
            assert 0,'Attribute '+str(cantFind)+' not found in '+str(self.jsonObject)
        elif(invalid):
            assert 0,'Attribute '+str(invalid)+' is invalid for '+str(self.jsonObject)

    def verifyAttribute(self,attr):

        if(attr not in self.jsonObject):
            self.myAssert(cantFind=attr)
        return True

    def initializeAllAttributes(self):
        self.X
        self.Y
        self.sex
        self.Id
        self.parents
        self.adoptiveParents
        self.mateKids
        self.mates
        self.diagnoses
        # self.affected
        self.age
        self.otherInfo
        self.divorcedList
        self.consanguinityList
        self.zygoticList
        self.noKids
        self.infertile
        self.dead
        self.ageOfDeath
        self.stillBirth
        self.ageOfStillBirth
        self.prematureDeath
        self.typeOfPrematureDeath
        self.unknownFamilyHistory
        self.proband
        self.consultand
        self.carrier
        self.pregnant
        self.surrogate
        self.donor

    def toString(self):
        return '\nId: '+str(self.Id)+', '+ \
        '[X: '+str(self.X)+', '+ \
        'Y: '+str(self.Y)+', '+ \
        'sex: '+str(self.sex)+', '+ \
        'parents: '+str([x.Id for x in self.parents])+', '+ \
        'adoptiveParents: '+str([x.Id for x in self.adoptiveParents])+', '+ \
        'mateKids: '+str([[x[0].Id, [y.Id for y in x[1]]] for x in self.mateKids])+', '+ \
        'mates: '+str([x.Id for x in self.mates])+', '+ \
        'diagnoses: '+str(self.diagnoses)+', '+ \
        'affected: '+str(self.affected)+', '+ \
        'age: '+str(self.age)+', '+ \
        'otherInfo: '+str(self.otherInfo)+', '+ \
        'divorcedList: '+str([x.Id for x in self.divorcedList])+', '+ \
        'consanguinityList: '+str([x.Id for x in self.consanguinityList])+', '+ \
        'zygoticList: '+str([x.Id for x in self.zygoticList])+', '+ \
        'noKids: '+str(self.noKids)+', '+ \
        'infertile: '+str(self.infertile)+', '+ \
        'dead: '+str(self.dead)+', '+ \
        'ageOfDeath: '+str(self.ageOfDeath)+', '+ \
        'stillBirth: '+str(self.stillBirth)+', '+ \
        'ageOfStillBirth: '+str(self.ageOfStillBirth)+', '+ \
        'prematureDeath: '+str(self.prematureDeath)+', '+ \
        'typeOfPrematureDeath: '+str(self.typeOfPrematureDeath)+', '+ \
        'unknownFamilyHistory: '+str(self.unknownFamilyHistory)+', '+ \
        'proband: '+str(self.proband)+', '+ \
        'consultand: '+str(self.consultand)+', '+ \
        'carrier: '+str(self.carrier)+', '+ \
        'pregnant: '+str(self.pregnant)+', '+ \
        'surrogate: '+str(self.surrogate)+', '+ \
        'donor: '+str(self.donor)+']\n'

    def X():
        doc = "The X property."
        def fget(self):
            if('_X' not in dir(self)):
                if(self.verifyAttribute('X')):
                    self._X = float(self.jsonObject['X'])
            return self._X
        def fset(self, value):
            self._X = value
        def fdel(self):
            del self._X
        return locals()
    X = property(**X())

    def Y():
        doc = "The Y property."
        def fget(self):
            if('_Y' not in dir(self)):
                if(self.verifyAttribute('Y')):
                    self._Y = float(self.jsonObject['Y'])
            return self._Y
        def fset(self, value):
            self._Y = value
        def fdel(self):
            del self._Y
        return locals()
    Y = property(**Y())

    def generation():
        doc = "The generation property."
        def fget(self):
            if('_generation' not in dir(self)):
                self._generation = 0
            return self._generation
        def fset(self, value):
            self._generation = value
        def fdel(self):
            del self._generation
        return locals()
    generation = property(**generation())

    def sex():
        doc = "The sex property."
        def fget(self):
            if('_sex' not in dir(self)):
                if(self.verifyAttribute('shapeName')):
                    if(self.jsonObject['shapeName'] == 'square'):
                        self._sex = 'male'
                    elif(self.jsonObject['shapeName'] == 'circle'):
                        self._sex = 'female'
                    elif(self.jsonObject['shapeName'] == 'diamond'):
                        self._sex = 'unknown'
                    else:
                        self.myAssert(invalid='shapeName')
            return self._sex
        def fset(self, value):
            self._sex = value
        def fdel(self):
            del self._sex
        return locals()
    sex = property(**sex())

    def Id():
        doc = "The Id property."
        def fget(self):
            if('_Id' not in dir(self)):
                if(self.verifyAttribute('Id')):
                    self._Id = int(self.jsonObject['Id'])
            return self._Id
        def fset(self, value):
            self._Id = value
        def fdel(self):
            del self._Id
        return locals()
    Id = property(**Id())

    def workForNegPerson(self, negID, mates):
        negPerson = idToPerson(negID,self.pedigree)
        if(len([x for x in mates if x!=negID])>12):
            assert 0,'wat is going on '+str([x for x in mates if x!=negID])
        other = idToPerson([x for x in mates if x!=negID][0],self.pedigree)

        # the person should have already been made, so just update
        # the shape name and the mateKids
        if(other.Id < 0):
            # then both parents are negative, so just make this person a female
            negPerson.sex = 'female'
        else:
            negPerson.sex = dict({'female':'male','male':'female','unknown':'female'})[other.sex]
            if(other.sex == 'unknown'):
                other.sex = 'male'
        negPerson.mateKids = [[other, other.children(negPerson)]]


    def parents():
        doc = "The parents property."
        def fget(self):
            if('_parents' not in dir(self)):
                if(self.verifyAttribute('parents')):
                    if(len([x for x in self.jsonObject['parents'].split(',') if len(x)>0]) == 0):
                        self._parents = []
                    else:
                        [parent1,parent2] = [x for x in self.jsonObject['parents'].split(',') if len(x)>0]
                        if(int(parent1)<0 and int(parent2)<0):
                            self.pedigree.specialCaseImpliedParents(idToPerson(int(parent1),self.pedigree),idToPerson(int(parent2),self.pedigree),self)
                        elif(int(parent1)<0):
                            self.workForNegPerson(parent1,self.jsonObject['parents'].split(','))
                        elif(int(parent2)<0):
                            self.workForNegPerson(parent2,self.jsonObject['parents'].split(','))
                        self._parents = [idToPerson(int(parent1),self.pedigree),idToPerson(int(parent2),self.pedigree)]

            return self._parents
        def fset(self, value):
            self._parents = value
        def fdel(self):
            del self._parents
        return locals()
    parents = property(**parents())

    def adoptiveParents():
        doc = "The adoptiveParents property."
        def fget(self):
            if('_adoptiveParents' not in dir(self)):
                if(self.verifyAttribute('adoptiveParents')):
                    if(len([x for x in self.jsonObject['adoptiveParents'].split(',') if len(x)>0]) == 0):
                        self._adoptiveParents = []
                    else:
                        [parent1,parent2] = [x for x in self.jsonObject['adoptiveParents'].split(',') if len(x)>0]
                        if(int(parent1)<0 and int(parent2)<0):
                            self.pedigree.specialCaseImpliedParents(idToPerson(int(parent1),self.pedigree),idToPerson(int(parent2),self.pedigree),self)
                        elif(int(parent1)<0):
                            self.workForNegPerson(parent1,self.jsonObject['adoptiveParents'].split(','))
                        elif(int(parent2)<0):
                            self.workForNegPerson(parent2,self.jsonObject['adoptiveParents'].split(','))
                        self._adoptiveParents = [idToPerson(int(parent1),self.pedigree),idToPerson(int(parent2),self.pedigree)]

            return self._adoptiveParents
        def fset(self, value):
            self._adoptiveParents = value
        def fdel(self):
            del self._adoptiveParents
        return locals()
    adoptiveParents = property(**adoptiveParents())

    def mateKids():
        doc = "The mateKids property."
        def fget(self):
            if(self.verifyAttribute('mateKids')):
                self._mateKids = []
                splitUp = [x for x in self.jsonObject['mateKids'][1:-1].split('~') if x != '']
                for x in splitUp:
                    # if the mateId is negative, then the work should have been handled by the parents/adopted parents function
                    # this is because a mate here means that someone has the implied person as a parent
                    self._mateKids.append([idToPerson(x.split(':')[0],self.pedigree), [idToPerson(y,self.pedigree) for y in [z for z in x.split(':')[1].split(',') if z!= '']]])
            return self._mateKids
        def fset(self, value):
            self._mateKids = value
        def fdel(self):
            del self._mateKids
        return locals()
    mateKids = property(**mateKids())

    def mates():
        doc = "The mates property."
        def fget(self):
            return [x[0] for x in self.mateKids]
        def fset(self, value):
            self._mates = value
        def fdel(self):
            del self._mates
        return locals()
    mates = property(**mates())

    def children(self,otherParent):
        if(otherParent not in self.mates):
            assert 0,'This person did not have kids with this other person'
        return [x[1] for x in self.mateKids if x[0] == otherParent][0]

    def diagnoses():
        doc = "The diagnoses property."
        def fget(self):
            if('_diagnoses' not in dir(self)):
                if(self.verifyAttribute('diagnosis')):
                    self._diagnoses = [x.split(',') for x in self.jsonObject['diagnosis'].split('~~') if len(x)>0]
            return self._diagnoses
        def fset(self, value):
            self._diagnoses = value
        def fdel(self):
            del self._diagnoses
        return locals()
    diagnoses = property(**diagnoses())

    def setAffectedFunctions(self,func):
        self.affectedFunction = func

    def affected():
        doc = "The affected property."
        def fget(self):
            # if we want to use different ways to define is a person is affected
            if('affectedFunction' in dir(self) and self.affectedFunction):
                return self.affectedFunction(self)
            assert 0,"no function to see if person is affected"

        def fset(self, value):
            self._affected = value
        def fdel(self):
            del self._affected
        return locals()
    affected = property(**affected())

    def age():
        doc = "The age property."
        def fget(self):
            if('_age' not in dir(self)):
                if(self.verifyAttribute('ageAtVisit')):
                    if(self.jsonObject['ageAtVisit'] == ''):
                        self._age = -1
                    else:
                        numbers = [s for s in self.jsonObject['ageAtVisit'].strip(';').strip('?').split(' ') if (isfloat(s[:-1]) or isfloat(s))]
                        if(len(numbers)==0):
                            numbers = [s for s in self.jsonObject['ageAtVisit'].strip(';').strip('?').split('-') if (isfloat(s[:-1]) or isfloat(s))]
                        if(not isfloat(numbers[0])):
                            self._age = float(numbers[0][:-1])+5
                        else:
                            self._age = float(numbers[0])
            return self._age
        def fset(self, value):
            self._age = value
        def fdel(self):
            del self._age
        return locals()
    age = property(**age())

    def otherInfo():
        doc = "The otherInfo property."
        def fget(self):
            if('_otherInfo' not in dir(self)):
                if(self.verifyAttribute('otherInfo')):
                    self._otherInfo = self.jsonObject['otherInfo']
            return self._otherInfo
        def fset(self, value):
            self._otherInfo = value
        def fdel(self):
            del self._otherInfo
        return locals()
    otherInfo = property(**otherInfo())

    def divorcedList():
        doc = "The divorcedList property."
        def fget(self):
            if('_divorcedList' not in dir(self)):
                if(self.verifyAttribute('divorcedList')):
                    self._divorcedList = [idToPerson(x,self.pedigree) for x in self.jsonObject['divorcedList'].split(',') if len(x)>0]
            return self._divorcedList
        def fset(self, value):
            self._divorcedList = value
        def fdel(self):
            del self._divorcedList
        return locals()
    divorcedList = property(**divorcedList())

    def consanguinityList():
        doc = "The consanguinityList property."
        def fget(self):
            if('_consanguinityList' not in dir(self)):
                if(self.verifyAttribute('consanguinityList')):
                    self._consanguinityList = [idToPerson(x,self.pedigree) for x in self.jsonObject['consanguinityList'].split(',') if len(x)>0]
            return self._consanguinityList
        def fset(self, value):
            self._consanguinityList = value
        def fdel(self):
            del self._consanguinityList
        return locals()
    consanguinityList = property(**consanguinityList())

    def isIdenticalZygote(self):
        if('zygoticType' not in dir(self)):
            if(self.verifyAttribute('zygoticType')):
                self.zygoticType = self.jsonObject['zygoticType']
        return self.zygoticType == 'monozygotic'

    def zygoticList():
        doc = "The zygoticList property."
        def fget(self):
            if('_zygoticList' not in dir(self)):
                if(self.verifyAttribute('zygoticList')):
                    self._zygoticList = [idToPerson(x,self.pedigree) for x in self.jsonObject['zygoticList'].split(',') if len(x)>0]
            return self._zygoticList
        def fset(self, value):
            self._zygoticList = value
        def fdel(self):
            del self._zygoticList
        return locals()
    zygoticList = property(**zygoticList())

    def noKids():
        doc = "The noKids property."
        def fget(self):
            if('_noKids' not in dir(self)):
                if(self.verifyAttribute('noChildrenList')):
                    self._noKids = len(self.jsonObject['noChildrenList'].split(',')) > 0
            return self._noKids
        def fset(self, value):
            self._noKids = value
        def fdel(self):
            del self._noKids
        return locals()
    noKids = property(**noKids())

    def infertile():
        doc = "The infertile property."
        def fget(self):
            if('_infertile' not in dir(self)):
                if(self.verifyAttribute('infertileList')):
                    self._infertile = len(self.jsonObject['infertileList'].split(',')) > 0
            return self._infertile
        def fset(self, value):
            self._infertile = value
        def fdel(self):
            del self._infertile
        return locals()
    infertile = property(**infertile())

    def dead():
        doc = "The dead property."
        def fget(self):
            if('_dead' not in dir(self)):
                if(self.verifyAttribute('dead')):
                    self._dead = self.jsonObject['dead'] == "true"
            return self._dead
        def fset(self, value):
            self._dead = value
        def fdel(self):
            del self._dead
        return locals()
    dead = property(**dead())

    def ageOfDeath():
        doc = "The ageOfDeath property."
        def fget(self):
            if('_ageOfDeath' not in dir(self)):
                if(self.verifyAttribute('ageOfDeath')):
                    if(self.jsonObject['ageOfDeath'] == ''):
                        self._ageOfDeath = -1
                    else:
                        if(self.jsonObject['ageOfDeath'] == 'infant'):
                            self.jsonObject['ageOfDeath'] = '0.8'
                        numbers = [s for s in self.jsonObject['ageOfDeath'].strip(';').strip('?').split() if (isfloat(s[:-1]) or isfloat(s))]
                        if(len(numbers)==0):
                            numbers = [s for s in self.jsonObject['ageOfDeath'].strip(';').strip('?').split('-') if (isfloat(s[:-1]) or isfloat(s))]
                        if(not isfloat(numbers[0])):
                            self._ageOfDeath = float(numbers[0][:-1])+5
                        else:
                            self._ageOfDeath = float(numbers[0])
            return self._ageOfDeath
        def fset(self, value):
            self._ageOfDeath = value
        def fdel(self):
            del self._ageOfDeath
        return locals()
    ageOfDeath = property(**ageOfDeath())

    def stillBirth():
        doc = "The stillBirth property."
        def fget(self):
            if('_stillBirth' not in dir(self)):
                if(self.verifyAttribute('stillBirth')):
                    self._stillBirth = self.jsonObject['stillBirth'] == "true"
            return self._stillBirth
        def fset(self, value):
            self._stillBirth = value
        def fdel(self):
            del self._stillBirth
        return locals()
    stillBirth = property(**stillBirth())

    def ageOfStillBirth():
        doc = "The ageOfStillBirth property."
        def fget(self):
            if('_ageOfStillBirth' not in dir(self)):
                if(self.verifyAttribute('ageOfStillBirth')):
                    if(self.jsonObject['ageOfStillBirth'] == ''):
                        self._ageOfStillBirth = -1
                    else:
                        self._ageOfStillBirth = float(self.jsonObject['ageOfStillBirth'])
            return self._ageOfStillBirth
        def fset(self, value):
            self._ageOfStillBirth = value
        def fdel(self):
            del self._ageOfStillBirth
        return locals()
    ageOfStillBirth = property(**ageOfStillBirth())

    def prematureDeath():
        doc = "The prematureDeath property."
        def fget(self):
            if('_prematureDeath' not in dir(self)):
                if(self.verifyAttribute('prematureDeath')):
                    self._prematureDeath = self.jsonObject['prematureDeath'] == "true"
            return self._prematureDeath
        def fset(self, value):
            self._prematureDeath = value
        def fdel(self):
            del self._prematureDeath
        return locals()
    prematureDeath = property(**prematureDeath())

    def typeOfPrematureDeath():
        doc = "The typeOfPrematureDeath property."
        def fget(self):
            if('_typeOfPrematureDeath' not in dir(self)):
                if(self.verifyAttribute('typeOfPrematureDeath')):
                    self._typeOfPrematureDeath = self.jsonObject['typeOfPrematureDeath']
            return self._typeOfPrematureDeath
        def fset(self, value):
            self._typeOfPrematureDeath = value
        def fdel(self):
            del self._typeOfPrematureDeath
        return locals()
    typeOfPrematureDeath = property(**typeOfPrematureDeath())

    def unknownFamilyHistory():
        doc = "The unknownFamilyHistory property."
        def fget(self):
            if('_unknownFamilyHistory' not in dir(self)):
                if(self.verifyAttribute('unknownFamilyHistory')):
                    self._unknownFamilyHistory = self.jsonObject['unknownFamilyHistory'] == "true"
            return self._unknownFamilyHistory
        def fset(self, value):
            self._unknownFamilyHistory = value
        def fdel(self):
            del self._unknownFamilyHistory
        return locals()
    unknownFamilyHistory = property(**unknownFamilyHistory())

    def proband():
        doc = "The proband property."
        def fget(self):
            if('_proband' not in dir(self)):
                if(self.verifyAttribute('proband')):
                    self._proband = self.jsonObject['proband'] == "true"
            return self._proband
        def fset(self, value):
            self._proband = value
        def fdel(self):
            del self._proband
        return locals()
    proband = property(**proband())

    def consultand():
        doc = "The consultand property."
        def fget(self):
            if('_consultand' not in dir(self)):
                if(self.verifyAttribute('consultand')):
                    self._consultand = self.jsonObject['consultand'] == "true"
            return self._consultand
        def fset(self, value):
            self._consultand = value
        def fdel(self):
            del self._consultand
        return locals()
    consultand = property(**consultand())

    def carrier():
        doc = "The carrier property."
        def fget(self):
            if('_carrier' not in dir(self)):
                if(self.verifyAttribute('carrier')):
                    self._carrier = self.jsonObject['carrier'] == "true"
            return self._carrier
        def fset(self, value):
            self._carrier = value
        def fdel(self):
            del self._carrier
        return locals()
    carrier = property(**carrier())

    def pregnant():
        doc = "The pregnant property."
        def fget(self):
            if('_pregnant' not in dir(self)):
                if(self.verifyAttribute('pregnant')):
                    self._pregnant = self.jsonObject['pregnant'] == "true"
            return self._pregnant
        def fset(self, value):
            self._pregnant = value
        def fdel(self):
            del self._pregnant
        return locals()
    pregnant = property(**pregnant())

    def surrogate():
        doc = "The surrogate property."
        def fget(self):
            if('_surrogate' not in dir(self)):
                if(self.verifyAttribute('surrogate')):
                    self._surrogate = self.jsonObject['surrogate'] == "true"
            return self._surrogate
        def fset(self, value):
            self._surrogate = value
        def fdel(self):
            del self._surrogate
        return locals()
    surrogate = property(**surrogate())

    def donor():
        doc = "The donor property."
        def fget(self):
            if('_donor' not in dir(self)):
                if(self.verifyAttribute('donor')):
                    self._donor = self.jsonObject['donor'] == "true"
            return self._donor
        def fset(self, value):
            self._donor = value
        def fdel(self):
            del self._donor
        return locals()
    donor = property(**donor())

class Pedigree:

    def __init__(self,allJSON):
        global allPeople
        allPeople = {}
        self.specialCase = {}
        self.allJSON = allJSON
        self.family = []
        self.probandDisease = None
        self.numbAffected = -1
        self.initAllPeople()

    def getPerson(self,ID):
        for person in self.family:
            if(person.Id == ID ):
                return person
        return None

    def specialCaseImpliedParents(self,parentA,parentB,child):
        [a,b] = sorted([parentA.Id,parentB.Id])
        key = str(a)+','+str(b)
        if(key not in self.specialCase):
            self.specialCase[key] = [child]
        else:
            self.specialCase[key].append(child)

    def setAffectedFunctions(self,func):

        def yes(person):
            return 'yes'
        def no(person):
            return 'no'
        def possibly(person):
            return 'possibly'

        for p in self.family:
            # print('\n\np.diagnoses '+str(p.diagnoses))
            # print('probandDisease: '+str(self.probandDisease))
            breakFlag = False
            for pDis in self.probandDisease:
                for curDis in p.diagnoses:
                    if(pDis[0] == curDis[0]):
                        # then this person has the same disease as the proband
                        if(pDis[3] == 'false'):
                            p.setAffectedFunctions(yes)
                        else:
                            p.setAffectedFunctions(possibly)
                        breakFlag = True
                        break
                if(breakFlag):
                    break
            if(not breakFlag):
                p.setAffectedFunctions(no)


    def initAllPeople(self):

        # initialize all of the shown parents
        for i,currentJSON in enumerate(self.allJSON):
            if(i==0):
                continue
            adding = addPerson(self,currentJSON)
            if(adding.proband == True):
                if not len(adding.diagnoses):
                    print('error: proband does not have a disease selected')
                    assert 0
                self.probandDisease = adding.diagnoses
                # print('adding the proband disease here  '+str(self.probandDisease))

        # initialize all of the implied parents
        for x in self.family:
            x.initializeAllAttributes()

        # handle the special cases
        for key in self.specialCase:

            [a,b] = key.split(',')
            parentA = idToPerson(a,self)
            parentB = idToPerson(b,self)
            children = self.specialCase[key]
            parentA.sex = 'female'
            parentB.sex = 'male'
            parentA.mateKids = [[parentB,children]]
            parentB.mateKids = [[parentA,children]]

        # tell each person what family they are a part of
        self.roots = [x for x in self.family if len(x.parents)==0]
        currentList = self.roots
        famCounter = 0
        while(len(currentList)>0):

            tempList = []
            for p in currentList:

                if(len(p.mates)==0):
                    continue

                for m in p.mates:
                    if(len(set(p.familyNumbers).intersection(m.familyNumbers))>0):
                        continue

                    p.familyNumbers.append(famCounter)
                    m.familyNumbers.append(famCounter)

                    for c in p.children(m):
                        c.familyNumbers.append(famCounter)
                        tempList.append(c)

                    famCounter += 1

            currentList = tempList

    def printAll(self):
        for x in self.family:
            print(x.toString())

    def studyID():
        doc = "The studyID property."
        def fget(self):
            if('_studyID' not in dir(self)):
                self._studyID = self.allJSON[0]['pedigreeId']
            return self._studyID
        def fset(self, value):
            self._studyID = value
        def fdel(self):
            del self._studyID
        return locals()
    studyID = property(**studyID())

    def ethnicity1():
        doc = "The ethnicity1 property."
        def fget(self):
            if('_ethnicity1' not in dir(self)):
                self._ethnicity1 = self.allJSON[0]['ethnicity1']
            return self._ethnicity1
        def fset(self, value):
            self._ethnicity1 = value
        def fdel(self):
            del self._ethnicity1
        return locals()
    ethnicity1 = property(**ethnicity1())

    def ethnicity2():
        doc = "The ethnicity2 property."
        def fget(self):
            if('_ethnicity2' not in dir(self)):
                self._ethnicity2 = self.allJSON[0]['ethnicity2']
            return self._ethnicity2
        def fset(self, value):
            self._ethnicity2 = value
        def fdel(self):
            del self._ethnicity2
        return locals()
    ethnicity2 = property(**ethnicity2())

    def inheritancePattern():
        doc = "The inheritancePattern property."
        def fget(self):
            if('_inheritancePattern' not in dir(self)):
                self._inheritancePattern = self.allJSON[0]['inheritancePattern']
            return self._inheritancePattern
        def fset(self, value):
            self._inheritancePattern = value
        def fdel(self):
            del self._inheritancePattern
        return locals()
    inheritancePattern = property(**inheritancePattern())

    def other():
        doc = "The other property."
        def fget(self):
            if('_other' not in dir(self)):
                self._other = self.allJSON[0]['other']
            return self._other
        def fset(self, value):
            self._other = value
        def fdel(self):
            del self._other
        return locals()
    other = property(**other())

