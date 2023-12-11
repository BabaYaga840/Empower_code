xavg1=xavg2=xavg3=[]
    n1=n2=n3=0
    for m in size:
        if m//thicc==maxx:
            xavg1.append(m)
            n1+=1
        if m//thicc==maxx2:
            xavg2.append(m)
            n2+=1
        if m//thicc==maxx3:
            xavg3.append(m)
            n3+=1
    xavgg1=np.mean(xavg1)
    xavgg2=np.mean(xavg2)
    xavgg3=np.mean(xavg3)
    if xavgg1<xavgg2 and xavgg1<xavgg3:
        volume=(xavgg2-xavgg1)/(xavgg3-xavgg1)
    if xavgg2<xavgg1 and xavgg2<xavgg3:
        volume=(xavgg1-xavgg2)/(xavgg3-xavgg2)
    if xavgg3<xavgg2 and xavgg3<xavgg1:
        volume=(xavgg2-xavgg3)/(xavgg1-xavgg3)