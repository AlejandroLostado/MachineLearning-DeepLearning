function aucfin = AUCcalc(X,y)
    for i=1:size(X,2)
        x=X(:,i);
        posidx=find(y>0);
        negidx=find(y<0);
        [p1,p2]=size(posidx);
        [n1,n2]=size(negidx);
        posout=repmat(x(posidx),n2,n1);
        negout=repmat(x(negidx)',p1,p2);
        rocmat=2*(negout<posout);
        rocmat(negout==posout)=1;
        aucfin(i)=sum(sum(rocmat))/(2*max(n1,n2)*max(p1,p2));	
    end 
end