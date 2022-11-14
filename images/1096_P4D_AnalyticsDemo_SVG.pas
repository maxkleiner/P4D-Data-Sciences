program VCL4Python_Py_AnalyticsDemo;
{
 Most programmers are familiar with the inexactness of floating point 
                                   calculations in a binary processor. 
 https://downloads.blaisepascal.eu/BlaisePascalMagazine_99_100_UK.pdf
 
 https://github.com/maxkleiner/DelphiVCL4Python
}

Const //PYHOME = 'C:\Users\max\AppData\Local\Programs\Python\Python36-32\';
PYHOME = 'C:\Users\breitsch\AppData\Local\Programs\Python\Python37-32\'; 
      
 VCLHOME = 
  'r"C:\users\max\appdata\local\programs\python\python36-32\lib\site-packages\delphivcl\win64\delphivcl.pyd"';

 
//{$I logo.p} 
PROCEDURE logo;
BEGIN
 writeln('');
 writeln('  .-------------------------------------------.');
 writeln('  | The Clean Python Bank Society         |');
 writeln('  | Task9                 1988-2022 M.K.     |');
 writeln('  .-------------------------------------------.');
 writeln('');
END; {procedure logo}

const LoadPy_VCLClass =

  'class MainForm(Form):                            '+LF+
  '                                                 '+LF+
  '  def __init__(self, owner):                     '+LF+
  '      self.Caption = "A VCL Form..."             '+LF+
  '      self.SetBounds(10, 10, 500, 400)           '+LF+
  '      self.Position = "poScreenCenter"           '+LF+
  '                                                 '+LF+
  '      self.lblHello = Label(self)                '+LF+
  '      self.lblHello.SetProps(Parent=self,        '+LF+
  '        Caption="Hello DelphiVCL maXbox4 Python")'+LF+
  '      self.lblHello.SetBounds(10, 10, 300, 24)   '+LF+
  '                                                 '+LF+
  '      self.OnClose = self.__on_form_close        '+LF+
  '                                                 '+LF+
  '                                                 '+LF+
  '  def __on_form_close(self, sender, action):     '+LF+
  '     action.Value = caFree                       ';
  
 
 STARTMAIN =
  'def main():                               '+LF+
  '  Application.Initialize()                '+LF+
  '  Application.Title = "Hello Python"      '+LF+
  '  Main = MainForm(Application)            '+LF+
  '  Main.Show()                             '+LF+
  '  FreeConsole()                           '+LF+
  '  Application.Run()                       '+LF+
  '  Main.Destroy()                          ';
  
  
 ANALYTICS='#from delphi_module import svg_image'+LF
    +'from io import StringIO'+LF
    +'import numpy as np'     +LF
    +'import pandas as pd'    +LF
    +'import matplotlib.pyplot as plt'+LF
    +'from sklearn.linear_model import BayesianRidge'+LF
    +'from sklearn.model_selection import RandomizedSearchCV, train_test_split'+LF
    +'from sklearn.preprocessing import PolynomialFeatures'+LF
    +'from sklearn.metrics import mean_squared_error, mean_absolute_error'+LF
    +'from sklearn.utils import parallel_backend'+LF
    +'parallel_backend('#39'threading'#39')'+LF
    +'import datetime'+LF
    +'import warnings'+LF
    +'warnings.filterwarnings("ignore")'+LF
    +''+LF
    +'confirmed_df = pd.read_csv('#39'https://raw.githubusercontent.com/CS'+
        'SEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_ti' +
        'me_series/time_series_covid19_confirmed_global.csv'#39')'+LF
    +''+LF
    +'parallel_backend('#39'threading'#39')'+LF
    +'cols = confirmed_df.keys()'+LF
    +'confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]'+LF
    +'dates = confirmed.keys()'+LF
    +'world_cases = []'+LF
    +''+LF
    +'for i in dates:'+LF
    +'    confirmed_sum = confirmed[i].sum()'+LF
    +'    world_cases.append(confirmed_sum)'+LF
    +'# window size'+LF
    +'window = 7'+LF
    +''+LF
    +'days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)'+LF
    +'world_cases = np.array(world_cases).reshape(-1, 1)'+LF
    +''+LF
    +'days_in_future = 10'+LF
    +'future_forcast = np.array([i for i in range(len(dates)+days_in_f' +
          'uture)]).reshape(-1, 1)' +LF
    +''+LF
    +'start = '#39'1/22/2020'#39+LF
    +'start_date = datetime.datetime.strptime(start, '#39'%m/%d/%Y'#39')'+LF
    +'future_forcast_dates = []' +LF
    +'for i in range(len(future_forcast)):'+LF
    +'  future_forcast_dates.append((start_date + datetime.timedelta' +
        '(days=i)).strftime('#39'%m/%d/%Y'#39'))'+LF
    +''+LF
    +'X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_c' +
        'onfirmed = train_test_split(days_since_1_22[50:], world_cases[50' +
        ':], test_size=0.05, shuffle=False)'+LF
    +''+LF
    +'# transform our data for polynomial regression'+LF
    +'bayesian_poly = PolynomialFeatures(degree=5)'+LF
    +'bayesian_poly_X_train_confirmed = bayesian_poly.fit_transform(X_' +
         'train_confirmed)'+LF
    +'bayesian_poly_X_test_confirmed = bayesian_poly.fit_transform(X_t' +
         'est_confirmed)'+LF
    +'bayesian_poly_future_forcast= bayesian_poly.fit_transform(future_forcast)'+LF
    +'' +LF
    +'# bayesian ridge polynomial regression'    +LF
    +'tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]'      +LF
    +'alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]'  +LF
    +'alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]'  +LF
    +'lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]' +LF
    +'lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]' +LF
    +'normalize = [True, False]'                 +LF
    +''                                          +LF
    +'bayesian_grid={'#39'tol'#39':tol,'#39'alpha_1'#39':alpha_1,'#39'alpha_2'#39':alp'+
    'ha_2, '#39'lambda_1'#39': lambda_1,'#39'lambda_2'#39' : lambda_2,'
    +'                  '#39'normalize'#39' : normalize}' +LF
    +''+LF
    +'bayesian = BayesianRidge(fit_intercept=True)' +LF
    +'bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, sc' +
        'oring='#39'neg_mean_squared_error'#39', cv=3, return_train_score=True, n'+
        '_jobs=-1, n_iter=40, verbose=1)'+LF
    +'bayesian_search.fit(bayesian_poly_X_train_confirmed, y_train_confirmed)'+LF
    +''+LF
    +'print(bayesian_search.best_params_)'+LF
    +''+LF
    +'bayesian_confirmed = bayesian_search.best_estimator_' +LF
    +  'test_bayesian_pred = bayesian_confirmed.predict(bayesian_poly_X_' +
       'test_confirmed)'+LF
    +'bayesian_pred = bayesian_confirmed.predict(bayesian_poly_future_forcast)'+LF
    +'print('#39'MAE:'#39',mean_absolute_error(test_bayesian_pred,y_test_confirmed))'+LF
    +'print('#39'MSE:'#39',mean_squared_error(test_bayesian_pred,y_test_confirmed))' +LF
    +'' +LF
    +'plt.plot(y_test_confirmed)'+LF
    +'plt.plot(test_bayesian_pred)'+LF
    +'plt.legend(['#39'Test Data'#39','#39'Bayesian Ridge Polynomial Predictions'#39'])'+LF
    +'' +LF
    +'figfile = StringIO()'+LF
    +'plt.savefig(figfile, format='#39'svg'#39')'+LF
    +'figdata_svg = figfile.getvalue()' +LF
    +'#svg_image.SvgText = figdata_svg'+LF
    +''+LF
    +'plt.show()';  
    
    
 ANALYTICSSVG='#from delphi_module import svg_image'+LF
    +'from io import StringIO'+LF
    +'import matplotlib.pyplot as plt'+LF
    +'import seaborn as sns'+LF
    +'df = sns.load_dataset("penguins")'+LF
    +'sns.pairplot(df, hue="species")'+LF
    +''+LF
    +'figfile = StringIO()'+LF
    +'plt.savefig(figfile, format="svg")'+LF
    +''+LF
    +'plt.show()'; 


procedure pyBank_VCL4Python;
var eg: TPythonEngine; sw: TStopWatch;
begin
eg:= TPythonEngine.Create(Nil);
  try
    eg.pythonhome:= PYHOME;
    eg.loadDLL;
    println('test import '+GetPythonEngine.EvalStr('__import__("decimal").Decimal(0.1)'));
    println('test import '+eg.EvalStr('__import__("decimal").Decimal(0.1)')); 
    writeln('') 
    //println(eg.EvalStr('__import__("faker").Faker()')); 
    sw:= TStopWatch.Create();
    sw.Start;
    eg.execStr('import importlib.machinery, importlib.util');
    eg.execStr('from decimal import Decimal, getcontext');
    //eg.execStr('from delphivcl import *');
    //eg.execStr('import delphivcl, os');
    //eg.execStr('loader= importlib.machinery.ExtensionFileLoader("DelphiVCL",'+VCLHOME+')')
    //println('loader test>>> '+eg.evalStr('loader'));
    //eg.execStr(LoadPy_VCLClass);
    //eg.execStr(STARTMAIN);
    //eg.execStr('main()');
    eg.execStr(ANALYTICSSVG);
    //println(eg.evalStr(''#39'MAE:'#39',mean_absolute_error(test_bayesian_pred,y_test_confirmed)'));
    //println(eg.evalStr(''#39'MSE:'#39',mean_squared_error(test_bayesian_pred, y_test_confirmed)'));
    
    //eg.execStr(pybankDEF)  
    //"Bank balance after 25 years = ", bank(25)
    // println('Bank balance after 25 years = '+
     //               eg.evalStr('bank(25)'));
    sw.Stop;
    //sw.ElapsedMilliseconds;
    writeln('Stop Analytics Tester1: '+sw.getValueStr)
  except
    eg.raiseError;
    writeln(ExceptionToString(ExceptionType, ExceptionParam));  
  finally
    eg.Free;
    sw.Free;
    sw:= Nil;
  end;
end;   

Const UPPERLIMIT = 100;
var cnt,n, precision: NativeUint;

begin //@main
  logo;
  pyBank_VCL4Python();
End.

ref: test import 0.1000000000000000055511151231257827021181583404541015625
test import 0.1000000000000000055511151231257827021181583404541015625

('MAE:', 40265042.8047044)
('MSE:', 1686906953419077.5)
Stop Analytics Tester1: 0:10:51.185


Exception: <class 'NameError'>: name 'svg_image' is not defined.

     0.03993872967323020890367053
     0.03993872967323020890367053
PY   0.03993872967323021

ref install python.exe -m pip install delphivcl

Exception: <class 'ValueError'>: DelphiVCL module not found. Try to reinstall the delphivcl package or check for support compatibility.

import sys
# the following is needed to use the newly allocated console!
sys.stdout = sys.stderr= open('CONOUT$', 'wt')
 

C:\maXbox\works2021\maxbox4>cd C:\Users\Max\AppData\Local\Programs\Python\Python36-32

C:\Users\Max\AppData\Local\Programs\Python\Python36-32>python.exe -m pip install
 delphivcl
Collecting delphivcl
  Downloading https://files.pythonhosted.org/packages/15/6b/32729afd53807fe5b890
dbe309da2ea3575f49f7fe21d15532672fc94763/delphivcl-0.1.29-cp36-cp36m-win32.whl (
2.1MB)
    100% |¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦| 2.1MB 539kB/s
Installing collected packages: delphivcl
Successfully installed delphivcl-0.1.29
You are using pip version 9.0.1, however version 21.3.1 is available.
You should consider upgrading via the 'python -m pip install --upgrade pip' command.

C:\Users\Max\AppData\Local\Programs\Python\Python36-32>

C:\Users\Max\AppData\Local\Programs\Python\Python36-32>python.exe -m pip uninsta
ll delphivcl
Uninstalling delphivcl-0.1.29:
  c:\users\max\appdata\local\programs\python\python36-32\lib\site-packages\delph
ivcl-0.1.29.dist-info\installer
  c:\users\max\appdata\local\programs\python\python36-32\lib\site-packages\delph
ivcl-0.1.29.dist-info\license.md
  c:\users\max\appdata\local\programs\python\python36-32\lib\site-packages\delph
ivcl-0.1.29.dist-info\metadata
  c:\users\max\appdata\local\programs\python\python36-32\lib\site-packages\delph
ivcl-0.1.29.dist-info\record
  c:\users\max\appdata\local\programs\python\python36-32\lib\site-packages\delph
ivcl-0.1.29.dist-info\top_level.txt
  c:\users\max\appdata\local\programs\python\python36-32\lib\site-packages\delph
ivcl-0.1.29.dist-info\wheel
  c:\users\max\appdata\local\programs\python\python36-32\lib\site-packages\delph
ivcl\__init__.py
  c:\users\max\appdata\local\programs\python\python36-32\lib\site-packages\delph
ivcl\__pycache__\__init__.cpython-36.pyc
  c:\users\max\appdata\local\programs\python\python36-32\lib\site-packages\delph
ivcl\__pycache__\__version__.cpython-36.pyc
  c:\users\max\appdata\local\programs\python\python36-32\lib\site-packages\delph
ivcl\__pycache__\moduledefs.cpython-36.pyc
  c:\users\max\appdata\local\programs\python\python36-32\lib\site-packages\delph
ivcl\__version__.py
  c:\users\max\appdata\local\programs\python\python36-32\lib\site-packages\delph
ivcl\moduledefs.py
  c:\users\max\appdata\local\programs\python\python36-32\lib\site-packages\delph
ivcl\win64\delphivcl.pyd
Proceed (y/n)? y
  Successfully uninstalled delphivcl-0.1.29
  
Error log:
  C:\Users\breitsch\Documents\Embarcadero\Studio\21.0\CatalogRepository\Delphi4PythonExporter-D104-1.2.1\Source;"C:\Program 
  Files\Streaming\maxbox4\python4delphi\Source";"C:\Program Files\Streaming\maxbox4\python4delphi\Source\vcl" -K00400000 
  -NBC:\Users\Public\Documents\Embarcadero\Studio\21.0\Dcp -NHC:\Users\Public\Documents\Embarcadero\Studio\21.0\hpp\Win32  Demo34.dpr   
[dcc32 Fatal Error] Demo34.dpr(7): F2048 Bad unit format: 'Unit1.dcu' - Expected version: 34.0, Windows Unicode(x86) Found version: 34.0, Windows Unicode(x64)
Failed
Elapsed time: 00:00:01.0
64bit build

procedure TDynamicDll.DoOpenDll(const aDllName : string);
{$IFDEF MSWINDOWS}
const
  LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = $00000100;
  LOAD_LIBRARY_DEFAULT_DIRS = $00001000;
Var
  ExceptMask: TFPUExceptionMask;
{$ENDIF}
begin
  if not IsHandleValid then
  begin
    FDllName := aDllName;
    {$IFDEF MSWINDOWS}
    ExceptMask := GetExceptionMask;
    try
      {$IFDEF FPC}
      FDLLHandle := LoadLibraryExA(PAnsiChar(AnsiString(GetDllPath+DllName)),
      {$ELSE}
      /// [dcc64 Error] PythonEngine.pas(2965): E2010 Incompatible types: 'PWideChar' and 'PAnsiChar'
      FDLLHandle := LoadLibraryEx(PWidechar(GetDllPath+DllName),
      {$ENDIF}
        0, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR or LOAD_LIBRARY_DEFAULT_DIRS);
    finally
      SetExceptionMask(ExceptMask);
    end;
    {$ELSE}
    //Linux: need here RTLD_GLOBAL, so Python can do "import ctypes"
    FDLLHandle := THandle(dlopen(PAnsiChar(AnsiString(GetDllPath+DllName)),
      RTLD_LAZY+RTLD_GLOBAL));
    {$ENDIF}
  end;
end;

function TMethodsContainer.AddMethod( AMethodName  : PAnsiChar;
                                      AMethod  : PyCFunction;
                                      ADocString : PAnsiChar ) : PPyMethodDef;
begin
  if FMethodCount = FAllocatedMethodCount then
    ReallocMethods;
  Result := Methods[ MethodCount ];
  Result^.ml_name  := AMethodName;
  Result^.ml_meth  := AMethod;
  Result^.ml_flags := METH_VARARGS;
  Result^.ml_doc   := ADocString;
  Inc( FMethodCount );
end;
