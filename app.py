from flask import Flask,render_template,request,jsonify
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/', methods=['GET', 'POST']) # To render Homepage
def home_page():
    return render_template('index.html')

# Doing EDA 
def age_convertor(Age):
    if ((Age<18) or (Age>70)):
        return 1
    else:
        return 2

def sex_convertor(Sex):
    if Sex == 'Male':
        return 1
    else:
        return 0 

def find_Alone(Sibsp):
    if Sibsp>0:
        return 1
    else:
        return 0              
    
    
def Parch_convertor(parch):
    if parch == 0:
        return 0
    elif parch ==1:
        return 1
    elif parch ==2:
        return 2
    else:
        return 3    


# def new_marriage_func(years_married):
#     if years_married < 3:
#         return 1
#     else:
#         return 0 

# def no_children_func(number_of_children):
#     if number_of_children == 0:
#         return 1
#     else:
#         return 0

# def rate_marriage_mapping(rate_marriage):
#     if rate_marriage == 'Very poor':
#         return 1
#     elif rate_marriage == 'Poor':
#         return 2
#     elif rate_marriage == 'Fair':
#         return 3
#     elif rate_marriage == 'Good':
#         return 4
#     else:
#         return 5

# def religious_mapping(religious):
#     if religious == 'Not':
#         return 1
#     elif religious == 'Mildly':
#         return 2
#     elif religious == 'Fairly':
#         return 3
#     else:
#         return 4           




@app.route('/titanic', methods=['POST'])  # This will be called from UI
def math_operation():
    if (request.method=='POST'):
        #operation=request.form['operation']
        Pclass=(request.form['Pclass'])
        Sex =request.form['Sex']
        Age =int(request.form['Age'])
        Sibsp= int(request.form['Sibsp'])
        Parch = int(request.form['Parch'])
        Fare = request.form['Fare']

        Sex_number = sex_convertor(Sex)
        Age = age_convertor(Age)
        Alone = find_Alone(Sibsp)
        parch = Parch_convertor(Parch)


        

   








        int_features = []
        int_features.append(Pclass)
        int_features.append(Sex_number)
        int_features.append(Age)
        int_features.append(Alone)
        int_features.append(parch)
        int_features.append(Fare)

        

        final_features = [np.array(int_features)]

        print(final_features)

        



        prediction = model.predict(final_features)

        print(prediction)

        # not_cheating = prediction[0][0]
        # cheating = prediction[0][1]

        if prediction==1:

            verdict = 'Survive'
            
        else:
            verdict = 'not Survive'  
            

        # prediction = [verdict, predict]   



        

        
        return render_template('results.html',result=verdict)  


if __name__ == '__main__':
    app.run(debug=True)          