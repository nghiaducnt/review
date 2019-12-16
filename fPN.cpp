#include "math.h"
#include "mex.h" 
#include <time.h>
#include <algorithm> 
#include <iostream> 


double objective(int m, int n, double *X, double *WH)
// this function is to calculate the KL objective function, size(X)=[m,n] 
{
	double obj = 0;
	for ( int i=0 ; i<m*n ; i++ )
		obj = obj + X[i]*log((X[i]+1e-10)/(WH[i]+1e-10))-X[i]+WH[i];
	return (obj);
}

void updateH(int m, int n, int r, double *sc_coeff, double *WH, double *X, double *W, double *H)
// this function is to update H(i,j)
{   
    
	int inner_iter, id_i, id_j = 5; // update H 5 times
    double grad=0, hessian=0, temp, id, s, lambda, vold, vnew, diff,d;

	for (int icount =0 ; icount<inner_iter ; icount++ )
    {
        for ( int j=0 ; j<n ; j++ )
        {
            for ( int i=0 ; i<r ; i++ )
            {
                // find gradient and hessian at H_{ij} 
                 grad=0;
                 hessian=0;
                 for (int idx=0 ; idx<m ; idx++ )
			     { id=idx*n;
                   id_j	= j+id;
                   id_i = i+id;
				   temp = X[id_j]/(WH[id_j]+1e-10);
				   grad = grad + W[id_i]*(1-temp); 
				   hessian = hessian + W[id_i]*W[id_i]*temp/(WH[id_j]+1e-10);   
			     }
                 // find Newton direction
                
                vold = H[j+i*n]; //H(i,j)
                s=std::max(1e-15,vold-grad/(hessian+1e-10));
                if (grad>0) 
                 { d=s-vold;
                   lambda=sc_coeff[j]*sqrt(hessian)*fabs(d);
                   if (lambda > 0.219224)
                     vnew=vold+1/(1+lambda)*d; // damp Newton step
                   else
                     vnew=s;
                 }
                else
                    vnew=s;
                // update the matrix WH
                diff=vnew-vold; 
                for (int idx=0 ; idx<m ; idx++ )
                {  id=idx*n;
                   id_j	= j+idx;
                   id_i = i+idx;
                   WH[id_j]=WH[id_j]+W[id_i]*diff;
                }
                // update H(i,j)
                H[j+i*n]=vnew;
            }
        }
    }
}

//////update W
void updateW(int m, int n, int r, double *sc_coeff, double *WH, double *X, double *W, double *H)
// update H(i,j)
{   
    
	int inner_iter = 5, id_i, id_j; // update H 5 times
    double grad=0, hessian=0, temp, id, d, s, lambda, vold, vnew, diff;

	for (int icount =0 ; icount<inner_iter ; icount++ )
    {
        for ( int i=0 ; i<m ; i++ )
        {
            for ( int j=0 ; j<r ; j++ )
            {
                // find gradient and hessian at H_{ij} 
                 grad=0;
                 hessian=0;
                for (int idx=0 ; idx<n ; idx++ )
			     { 
                   id_i	= i*n+idx;
                   id_j = j*n+idx;
				   temp = (X[id_i])/(WH[id_i]+1e-10);
				   grad = grad + H[id_j]*(1-temp); 
				   hessian = hessian + H[id_j]*H[id_j]*temp/(WH[id_i]+1e-10);   
			     }
                 // find Newton direction
                
                vold = W[j+i*r]; //W(i,j)
                s=std::max(1e-15,vold-grad/(hessian+1e-10));
                if (grad>0) 
                 { d=s-vold;
                   lambda=sc_coeff[i]*sqrt(hessian)*fabs(d);
                   if (lambda > 0.219224)
                     vnew=vold+1/(1+lambda)*d; // damp Newton step
                   else
                     vnew=s;
                 }
                else
                    vnew=s;
                // update the matrix WH
                diff=vnew-vold; 
                for (int idx=0 ; idx<n; idx++ )
                {  id=i*n;
                   id_j	= j*n+idx;
                   id_i = i*n+idx;
                   WH[id_i]=WH[id_i]+H[id_j]*diff;
                }
                // update W(i,j)
                W[j+i*r]=vnew;
            }
        }
    }
}


void mainupdate(int m, int n, int r, int maxiter, double maxtime, double *X, double *W, double *H, double *obj, double *time)
{
    double total = 0, timestart;
    double *WH = (double *)malloc(sizeof(double)*m*n); //N row, M col?
    double *col=(double *)malloc(sizeof(double)*n);
    double *row=(double *)malloc(sizeof(double)*m);
    int idx=0, iter=0, id;
    timestart = clock();
    for ( int j=0 ; j<n ; j++ ) //j : row
     for ( int i=0 ; i<m ; i++ ) // i : col
		{   // find WH(i,j)
			WH[idx] = 0;
			for (int k=0 ; k<r ; k++ ) //confuse?
				WH[idx] += W[k+i*m]*H[j+k*n]; // size of W and H? i*n? and k*m?
            idx += 1; 
		}
    // find col(k) 
    for (int j=0; j<n; j++)
    {   
        col[j]=sqrt(X[j]);
     for (int i=1; i<m; i++)    
     {   id=j+i*n; //traverse col of X?
         if (X[id]>0 && col[j]>0)
              col[j]=std::min(col[j],X[id]);
        
        else if (X[id]==0 && col[j]>0)
              col[j]=sqrt(X[id]);
      //what is the else condition, we should have else condition to catch all.
                
     }
     col[j]=1/col[j];   //if the for loop above cannot assign value to col[j], and its original value is 0-> exception
    } 
    // find row(k) 
    for (int i=0; i<m; i++)
    {   id=i*n;
        row[i]=sqrt(X[id]);
     for (int j=1; j<n; j++)    
     {   idx=id+j;//traverse row of X
         if (X[idx]>0 && row[i]>0)
              row[i]=std::min(row[i],X[idx]);
        
        else if (X[idx]==0 && row[i]>0)
              row[i]=sqrt(X[idx]);
        //what is the else condition, we should have else condition to catch all.  
     }
     row[i]=1/row[i];   
    }
    // Xt=X';
  
   	time[iter] = (clock()-timestart)/CLOCKS_PER_SEC;
    obj[iter]=objective(m, n, X, WH);
    for ( int iter=0 ; iter<maxiter ; iter++)
    {
        timestart = clock();
        // update H
         updateH(m, n, r, col, WH, X, W, H);
        // update W
         updateW( m, n, r, row, WH, X, W, H);
         time[iter +1] =time[iter] +(clock()-timestart)/CLOCKS_PER_SEC;
         obj[iter+1]=objective(m, n, X, WH);
         if (time[iter +1]>maxtime)
         {break;
         }
    }
    
  
	free(WH);
    free(col);
    free(row);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *xValues;
	int i,j;
	double avg;
	double *X, *W, *H;
	double *time_sequence = NULL, *obj_sequence = NULL;

	int n,m,r, maxiter, maxtime;
	double *outArray;
    
    X = mxGetPr(prhs[0]);
	m = mxGetM(prhs[0]);
	n = mxGetN(prhs[0]);

	r = mxGetScalar(prhs[1]);
	maxiter = mxGetScalar(prhs[2]);
    maxtime =  mxGetScalar(prhs[3]);
            
    W = mxGetPr(prhs[4]);
  	H = mxGetPr(prhs[5]);
    
    plhs[2] = mxCreateDoubleMatrix(1,maxiter,mxREAL);
	obj_sequence = mxGetPr(plhs[2]);
    
	plhs[3] = mxCreateDoubleMatrix(1,maxiter,mxREAL);
	time_sequence = mxGetPr(plhs[3]);
    //How big is W, X , H?
    mainupdate(m,n, r,maxiter,maxtime,X,W,H,obj_sequence, time_sequence);
    
    plhs[0] = mxCreateDoubleMatrix(m,r,mxREAL);
    	outArray=mxGetPr(plhs[0]);
	for ( int i=0 ; i<r*m ; i++ )
		outArray[i] = W[i];
    
	plhs[1] = mxCreateDoubleMatrix(r,n,mxREAL);
	outArray=mxGetPr(plhs[1]);
	for ( int i=0 ; i<r*n ; i++ )
		outArray[i] = H[i];

	return;

        

}


