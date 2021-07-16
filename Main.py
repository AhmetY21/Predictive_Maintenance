#Main

import pandas as pd
import feature_extract
import signal_class
import markov




def main():
    try:
        fileName=sys.argv[1]

    except Exception as e:
        print("You must provide a valid filename as parameter")
        raise

    table=m2_converter(sys.argv[1],sys.argv[2],sys.argv[3])
    breakdowne=['Time.6', 'Time.4', 'Time.1'] #example

    table=lab_break(table,1,breakdowne,widget=False)
    names=table['Name'].values
    table.drop('Name',axis=1,inplace=True)


    X=table.copy()
    X.drop('label',axis=1,inplace=True)
    y=table['label'].values


    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.25, random_state=58)

    clf = LogisticRegression(random_state=0,multi_class='auto',solver='liblinear')
    gnb = GaussianNB()
    sf=svm.SVC()
    dt = DecisionTreeClassifier(random_state=0,max_depth=3)

    benchmark(X_train,y_train,X_test,y_test,"Base")


    P=transition_matrix(mark)
    state=np.array([[0.3, 0.2, 0.1,0.4]])
    mark=np.random.randint(4, size=len(table))


    Steady_State(state,P,20)

if __name__ == "__main__":
    main()
