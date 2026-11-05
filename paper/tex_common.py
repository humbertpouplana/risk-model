import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
R=json.load(open(ROOT/'results'/'results.json'))
D={'p4':R}
PAR=R['params']
E=lambda v:f'{v:,.0f}\\,\\texteuro{{}}'.replace(',','{,}') if False else f'{v:,.0f}\\,\\texteuro{{}}'
def n0(v): return f'{v:,.0f}'
def n1(v): return f'{v:,.1f}'
def n2(v): return f'{v:,.2f}'
def n3(v): return f'{v:.3f}'
def pc(v,d=1): return f'{v*100:.{d}f}\\%'
PRE=r'''\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern,textcomp}
\usepackage[margin=2.4cm]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx,booktabs,array,float}
\usepackage[hidelinks]{hyperref}
\usepackage{microtype}
\setlength{\parskip}{3pt}
\newtheorem{proposition}{Proposition}
\newtheorem{definition}{Definition}
\renewcommand{\arraystretch}{1.1}
\begin{document}
'''
import re
def _fixlabels(t): return re.sub(r'\\label\{([^}]*)\}\\caption\{(.*?)\}\\end\{figure\}',lambda m:'\\caption{'+m.group(2)+'}\\label{'+m.group(1)+'}\\end{figure}',t,flags=re.S)
def make(title,subtitle,abstract,body,refs):
    return _fixlabels(PRE+r'\title{\bfseries '+title+r'\\[2pt]\large '+subtitle+'}\n\\author{Humbert Pou Plana}\n\\date{November 2026}\n\\maketitle\n'
            +'\\begin{abstract}\n'+abstract+'\n\\end{abstract}\n'+body+'\n\\begin{thebibliography}{9}\n'+refs+'\n\\end{thebibliography}\n\\end{document}\n')
def fig(name,cap,w='0.92'):
    return '\\begin{figure}[H]\\centering\\includegraphics[width='+w+'\\linewidth]{fig/'+name+'.pdf}\\caption{'+cap+'}\\end{figure}\n'
def esc(s): return s.replace('&','\\&').replace('%','\\%').replace('_','\\_').replace('#','\\#')

TR={'Deuda/ingresos':'Debt-to-income','Hora de solicitud':'Application hour','Consultas recientes':'Recent inquiries','Cuota/ingresos':'Instalment-to-income','Flujo/ingresos':'Inflow-to-income','Edad':'Age','Ingresos':'Income','Importe':'Loan size','Plazo':'Term',
 'Antigüedad laboral':'Employment tenure','Contrato temporal':'Temporary contract','Datos bancarios':'Bank data','Ingresos en cuenta':'Bank inflow','Descubierto':'Overdraft','Historial crediticio':'Credit history length','Préstamos previos':'Previous loans',
 'Tiempo en formulario':'Form time','Email gratuito':'Free email','Canal':'Channel','Dispositivo':'Device','Importe/ingresos':'Amount-to-income','Perfil':'Profile','Deuda y solicitud':'Debt and application','Comportamiento':'Behaviour',
 'sem_marca':'sem_brand','sem_generico':'sem_generic','afiliados':'affiliates','directo':'direct','ingresos bajos':'low income','ingresos medios':'mid income','ingresos altos':'high income','importe pequeño':'small loan','importe medio':'medium loan','importe grande':'large loan',
 'con datos bancarios':'with bank data','sin datos bancarios':'without bank data','movil':'mobile','escritorio':'desktop','Datos bancarios (tiene_bancos)':'Linked bank data'}
def EN(s):
    return TR.get(s,s)
