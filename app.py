import flask
import pickle
import xgboost
import pandas as pd
#'Terit', 'UtsteinCohort', 'Vasc', 'InitialRhyth', 'Age2', 'normalECG','ste', 'std', 'rbbb', 'TTE'
app = flask.Flask(__name__, template_folder='templates')
app.static_folder = 'static'




# usingCalibrated = True
# if usingCalibrated == True:
#     model = pickle.load(open("./model/webapp_Model_Calibrated.pkl", "rb"))
# else:
model = pickle.load(open("./model/webapp_XGB_2022.pkl", "rb"))


print('testing Testing', flush=True)
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        #print('Getting', flush=True)
        # Just render the initial form, to get input
        Terit=UtsteinCohort=Vasc=InitialRhyth=Age2=normalECG=ste=rbbb=TTE = True
        res_dict = {True:'checked', False:'unchecked'}
        return flask.render_template('main.html',original_input={'Terit':Terit, 'UtsteinCohort':UtsteinCohort, 'Vasc':Vasc,'InitialRhyth':InitialRhyth, 'Age2':Age2, 'normalECG':normalECG, 'ste':ste, 'rbbb':rbbb,'TTE':TTE,},result= 0.0, resultN = 0.0, resd = [res_dict[x] for x in [Terit,UtsteinCohort,Vasc,InitialRhyth,Age2,normalECG,ste,rbbb,TTE]], bar_color = "transparent" )

    if flask.request.method == 'POST':
        #print('Post', flush=True)
        # Extract the input

        Terit = flask.request.form.get('Terit') != None
        UtsteinCohort = flask.request.form.get('UtsteinCohort')!= None
        Vasc = flask.request.form.get('Vasc')!= None
        InitialRhyth = not flask.request.form.get('InitialRhyth')!= None
        Age2 = flask.request.form.get('Age2')!= None
        #print('Age2: ', Age2)
        if Age2==False:
            AgeRange=2 # This is because Age2 can be 0,1,2 whereas everythign else can be 0,1
        else: AgeRange=Age2
        #print('Age2: ', AgeRange)
        normalECG = flask.request.form.get('normalECG')!= None
        ste = flask.request.form.get('ste')!= None
        rbbb = flask.request.form.get('rbbb')!= None
        TTE = not flask.request.form.get('TTE')!= None
        #print(TTE,flush=True)
        # Make DataFrame for model
        input_variables = pd.DataFrame([[Terit, UtsteinCohort, Vasc, InitialRhyth, AgeRange, normalECG,ste, rbbb, TTE]],
                                       columns=['Terit', 'UtsteinCohort', 'Vasc', 'InitialRhyth', 'Age2', 'normalECG','ste', 'rbbb', 'TTE'],
                                       dtype=float)

        # Get the model's prediction
        prediction = model.predict_proba(input_variables, ntree_limit=0, validate_features=True, iteration_range=[0,200])[0][0]
        #prediction = model.predict_proba(input_variables)[0][0] ## This is for the calibrated model
        # Render the form again, but add in the prediction and remind user
        # of the values they input before

        if 1 - prediction >= 0.7:
            culp_c = '#56CC9D'
        elif 1-prediction <= 0.3:
            culp_c = '#FF7851'
        else: culp_c = '#FFCE67'

        res_dict = {True:'checked', False:'unchecked'}
        return flask.render_template('main.html', original_input={'Terit':Terit, 'UtsteinCohort':UtsteinCohort, 'Vasc':Vasc,'InitialRhyth':not InitialRhyth, 'Age2':Age2, 'normalECG':normalECG, 'ste':ste, 'rbbb':rbbb,'TTE':not TTE,},result= 1-prediction, resultN = prediction, resd = [res_dict[x] for x in [Terit,UtsteinCohort,Vasc,not InitialRhyth,Age2,normalECG,ste,rbbb,not TTE]], bar_color = culp_c )

if __name__ == '__main__':
    app.run(debug=False)