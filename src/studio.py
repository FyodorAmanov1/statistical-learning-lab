"""Chronological nonlinear volatility forecasts, shift diagnostics and conformal bands."""
import numpy as np
try:
    from .common import finalize,metric
except ImportError:
    from common import finalize,metric


def generate(shift=False,seed=73,n=2000):
    rng=np.random.default_rng(seed);z=rng.normal(size=n);r=np.zeros(n);v=np.ones(n)*.8
    for t in range(1,n):
        shock=shift and t>=1450
        v[t]=(.12 if shock else .025)+(.16 if shock else .1)*r[t-1]**2+.84*v[t-1]
        r[t]=np.sqrt(v[t])*z[t]
    # Forecast next-day realized variance from features available at the current close.
    x=[];y=[];indices=[]
    for t in range(20,n-1):
        past=r[t-19:t+1]
        x.append([r[t]**2,np.mean(past[-5:]**2),np.mean(past**2),r[t],np.mean(abs(past)),np.mean(past[-5:])])
        y.append(r[t+1]**2);indices.append(t+1)
    return np.array(x),np.array(y),np.array(indices),v


def ridge_fit(x,y,alpha):
    mean=x.mean(axis=0);std=x.std(axis=0);std=np.where(std<1e-12,1,std)
    a=np.column_stack([np.ones(len(x)),(x-mean)/std]);penalty=np.eye(a.shape[1])*alpha;penalty[0,0]=0
    beta=np.linalg.solve(a.T@a+penalty,a.T@y)
    return lambda future: np.maximum(0,np.column_stack([np.ones(len(future)),(future-mean)/std])@beta)


def neural_fit(x,y,seed=53,epochs=350):
    """Small tanh network trained on log1p variance; all normalization is training-only."""
    mean=x.mean(axis=0);std=np.maximum(x.std(axis=0),1e-12);a=(x-mean)/std
    target=np.log1p(y);ym=target.mean();ys=target.std();target=(target-ym)/ys
    rng=np.random.default_rng(seed);w=rng.normal(0,.2,(a.shape[1],12));b=np.zeros(12);out=rng.normal(0,.2,12);intercept=0.
    # Full-batch gradient descent, fixed training budget; validation never contributes gradients.
    for _ in range(epochs):
        hidden=np.tanh(a@w+b);err=hidden@out+intercept-target;derivative=2*err/len(a)
        dh=derivative[:,None]*out*(1-hidden**2)
        dw=a.T@dh+1e-3*w;db=dh.sum(axis=0);do=hidden.T@derivative+1e-3*out;di=derivative.sum()
        w-=.025*dw;b-=.025*db;out-=.025*do;intercept-=.025*di
    return lambda future:np.maximum(0,np.expm1(np.clip((np.tanh((future-mean)/std@w+b)@out+intercept)*ys+ym,-20,20)))


def conformal_quantile(errors,coverage=.9):
    rank=min(len(errors),int(np.ceil((len(errors)+1)*coverage)))
    return float(np.sort(errors)[rank-1])


def experiment():
    cases=[]
    for shift in [False,True]:
        x,y,time,true_variance=generate(shift)
        # Cutoffs in target time; one-target embargo between fitting/tuning/calibration/test partitions.
        train=time<1000;validation=(time>=1001)&(time<1200);calibration=(time>=1201)&(time<1450);test=time>=1451
        train_x=x[train];train_y=y[train];validation_y=y[validation]
        candidates=[.01,1,10,100];scores=[]
        for alpha in candidates:
            predict=ridge_fit(train_x,train_y,alpha);scores.append(float(np.mean((predict(x[validation])-validation_y)**2)))
        alpha=candidates[int(np.argmin(scores))];ridge=ridge_fit(train_x,train_y,alpha);neural=neural_fit(train_x,train_y)
        constant=float(train_y.mean());models={'Training mean':lambda a:np.full(len(a),constant),'Lagged square':lambda a:a[:,0],'Ridge':ridge,'Nonlinear MLP':neural}
        predictions={name:predict(x[test]) for name,predict in models.items()};truth=y[test];times=time[test]
        radius=conformal_quantile(abs(y[calibration]-ridge(x[calibration])));lower=np.maximum(0,predictions['Ridge']-radius);upper=predictions['Ridge']+radius
        coverage=float(np.mean((truth>=lower)&(truth<=upper)));mae=lambda a:float(np.mean(abs(a-truth)))
        # Causal expanding-window refit: fixed hyperparameter, fitting includes only targets strictly before the block.
        rolling=np.zeros(len(truth))
        for start in range(0,len(times),60):
            now=times[start];allowed=time<now
            rolling[start:start+60]=ridge_fit(x[allowed],y[allowed],alpha)(x[test][start:start+60])
        predictions['Expanding ridge']=rolling
        qlike=lambda p:float(np.mean(np.log(np.maximum(p,1e-5))+truth/np.maximum(p,1e-5)))
        stride=3;shown=np.arange(0,len(truth),stride)
        blocks=[slice(i,min(i+60,len(truth))) for i in range(0,len(truth),60)]
        block_x=[int(times[b.start]) for b in blocks]
        cases.append({'id':'shift' if shift else 'stationary','label':'Volatility regime break' if shift else 'Stable regime','kicker':'TIME-SERIES ML / CAUSAL EVALUATION','title':'A forecast is only useful out of sample.',
          'description':'Predict next-day realized variance from six causal return features. Compare a small neural network, regularized regression, persistence and an expanding-window adaptation under a controlled volatility shock.',
          'metrics':[metric('Frozen ridge MAE',mae(predictions['Ridge'])),metric('Expanding ridge MAE',mae(rolling)),metric('Nominal interval coverage',90,'%'),metric('Observed test coverage',coverage*100,'%')],
          'charts':[{'title':'Held-out realized variance and frozen prediction interval · every third target','x':times[shown].tolist(),'xlabel':'Target day index','series':[{'name':'Realized squared return','values':truth[shown].tolist()},{'name':'Frozen ridge','values':predictions['Ridge'][shown].tolist()},{'name':'90% lower','values':lower[shown].tolist()},{'name':'90% upper','values':upper[shown].tolist()}]},
            {'title':'Error through time · non-overlapping blocks','x':block_x,'xlabel':'First target in block','series':[{'name':name,'values':[float(np.mean(abs(p[b]-truth[b]))) for b in blocks]} for name,p in predictions.items()]},
            {'title':'Calibration transport diagnostic','x':block_x,'xlabel':'First target in block','series':[{'name':'Observed interval coverage / %','values':[float(np.mean((truth[b]>=lower[b])&(truth[b]<=upper[b]))*100) for b in blocks]},{'name':'Nominal / %','values':[90]*len(blocks)}]}],
          'table':{'columns':['Model','Test MAE','Test RMSE','QLIKE (lower is better)'],'rows':[[name,mae(p),float(np.sqrt(np.mean((p-truth)**2))),qlike(p)] for name,p in predictions.items()]},
          'method':[f'Seed 73: simulated GARCH-like returns. Features at close t: squared return, 5/20-day realized variance, return, 20-day mean absolute return and 5-day mean return. Label: r² at t+1. Shift starts at target day 1450 when enabled.',f'Training targets <1000; validation 1001–1199 chooses ridge penalty {alpha}; calibration 1201–1449 sets a finite-sample absolute-residual quantile; test starts 1451. Scaling is fitted on training only. Embargoed boundaries prevent target overlap.', 'A 6→12→1 tanh MLP trains for 350 full-batch epochs on log1p variance, with fixed settings and training-only normalization. No claim that this architecture is optimal is made.',f'Frozen ridge interval radius = {radius:.6f}; lower endpoint clipped at zero. Expanding ridge refits every 60 targets, uses only earlier observed labels and keeps the previously selected penalty. It does not reuse the frozen interval as an adaptive guarantee.'],
          'limitations':['Synthetic heteroscedastic data tests a known mechanism; it is not a historical market backtest or evidence of alpha. Squared returns are noisy variance proxies.','Serial dependence and deliberate distribution shift violate exchangeability. Split-conformal intervals here are diagnostics; 90% empirical coverage is not guaranteed under these conditions.','Mean absolute error, RMSE and QLIKE can rank models differently. QLIKE uses a 10⁻⁵ positivity floor, disclosed because near-zero forecasts can dominate it.','Only one process seed is shown. Hyperparameters are selected without test data, but broad model selection uncertainty and transaction costs are not quantified.'],'reproduce':'python src/studio.py --out docs/study.json'})
    return {'schema_version':1,'slug':'statistical-learning-lab','title':'RegimeLab','subtitle':'Forecasting, uncertainty and adaptation when the data changes.','accent':'#78b9ff','seed':73,'cases':cases,
      'provenance':'Seeded synthetic returns; strictly ordered training, tuning, calibration and held-out targets. All reported models are computed locally with NumPy.',
      'references':[{'title':'A gentle introduction to conformal prediction','url':'https://arxiv.org/abs/2107.07511'},{'title':'Forecasting: principles and practice · time-series cross-validation','url':'https://otexts.com/fpp3/tscv.html'}]}


if __name__=='__main__':finalize(experiment())
