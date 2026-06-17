// mf_verify.cpp — verify a matrix-free C3D8 element operator reproduces the assembled A0. Tooling/APHYSICAL.
// The row236 grid is a homogeneous 0.1mm Cartesian voxel C3D8 mesh (EPU40 linear, E=8 MPa, nu=0.49), so EVERY
// element has the SAME 24x24 K_e. Matrix-free apply  y = sum_e G_e^T K_e G_e x  (G_e = 8-node gather, fixed
// DOFs treated as 0 and not scattered) must equal the assembled A0*x for any x. If it matches to roundoff, the
// analytic K_e is exact and we can drop the 33M-block BCSR stream in the GMG fine SpMV.
//   build: clang++ -std=c++17 -O3 -DNDEBUG mf_verify.cpp -o mf_verify
//   run:   ./mf_verify m2.mat coordmap.bin [E nu h]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#include <array>
#include <chrono>
#include <algorithm>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif
// Cross-platform by default; Apple Accelerate (AMX-backed cblas_dgemm for the batched apply) layered on
// macOS unless CCX_GMG_NOACCEL is set at compile time -> lets the portable reference build anywhere.
#if defined(__APPLE__) && !defined(CCX_GMG_NOACCEL)
  #define CCX_GMG_ACCEL 1
  #include <Accelerate/Accelerate.h>
#else
  #define CCX_GMG_ACCEL 0
#endif

// ---- analytic 24x24 C3D8 stiffness for an axis-aligned cube of side h, isotropic (E,nu) ----
// corner ordering c=0..7 at physical offset (c&1, (c>>1)&1, (c>>2)&1)*h ; local dof = 3*c + comp.
static void c3d8_Ke(double E, double nu, double h, double Ke[24][24]) {
    for (int i=0;i<24;i++) for (int j=0;j<24;j++) Ke[i][j]=0.0;
    // isotropic D (6x6), Voigt [xx,yy,zz,xy,xz,yz]
    double lam = E*nu/((1.0+nu)*(1.0-2.0*nu)), mu = E/(2.0*(1.0+nu));
    double D[6][6]={{0}};
    for(int i=0;i<3;i++){ for(int j=0;j<3;j++) D[i][j]=lam; D[i][i]+=2.0*mu; }
    D[3][3]=D[4][4]=D[5][5]=mu;
    // corner natural signs s=2*offset-1
    double sx[8],sy[8],sz[8];
    for(int c=0;c<8;c++){ sx[c]=2*((c)&1)-1; sy[c]=2*((c>>1)&1)-1; sz[c]=2*((c>>2)&1)-1; }
    double g=1.0/std::sqrt(3.0); double gp[2]={-g,g};
    double dxi=2.0/h, detJ=(h/2.0)*(h/2.0)*(h/2.0);   // J = (h/2)I for the cube
    // gradients at the 8 Gauss points
    double bx[8][8],by[8][8],bz[8][8];   // [gp][node]
    for(int q=0;q<8;q++){ int a=q&1,b=(q>>1)&1,cc=(q>>2)&1; double xi=gp[a],et=gp[b],ze=gp[cc];
        for(int c=0;c<8;c++){
            bx[q][c]=0.125*sx[c]*(1+sy[c]*et)*(1+sz[c]*ze)*dxi;
            by[q][c]=0.125*sy[c]*(1+sx[c]*xi)*(1+sz[c]*ze)*dxi;
            bz[q][c]=0.125*sz[c]*(1+sx[c]*xi)*(1+sy[c]*et)*dxi;
        } }
    // B-bar: element-averaged dilatational gradients (cube -> equal-weight average over the 8 GPs)
    double bbx[8]={0},bby[8]={0},bbz[8]={0};
    for(int c=0;c<8;c++){ for(int q=0;q<8;q++){ bbx[c]+=bx[q][c]; bby[c]+=by[q][c]; bbz[c]+=bz[q][c]; }
        bbx[c]/=8.0; bby[c]/=8.0; bbz[c]/=8.0; }
    for(int q=0;q<8;q++){
        double B[6][24]={{0}};
        for(int c=0;c<8;c++){ int d=3*c;
            double b1=bx[q][c],b2=by[q][c],b3=bz[q][c];
            double dx=(bbx[c]-b1)/3.0, dy=(bby[c]-b2)/3.0, dz=(bbz[c]-b3)/3.0;  // (b_bar - b)/3
            // normal rows (B-bar): B + dilatational correction
            B[0][d+0]=b1+dx; B[0][d+1]=dy;    B[0][d+2]=dz;
            B[1][d+0]=dx;    B[1][d+1]=b2+dy; B[1][d+2]=dz;
            B[2][d+0]=dx;    B[2][d+1]=dy;    B[2][d+2]=b3+dz;
            // shear rows: standard (GP-local gradients)
            B[3][d+0]=b2; B[3][d+1]=b1;
            B[4][d+0]=b3; B[4][d+2]=b1;
            B[5][d+1]=b3; B[5][d+2]=b2;
        }
        // Ke += detJ * B^T D B  (w=1)
        double DB[6][24];
        for(int i=0;i<6;i++)for(int j=0;j<24;j++){ double s=0; for(int k=0;k<6;k++) s+=D[i][k]*B[k][j]; DB[i][j]=s; }
        for(int i=0;i<24;i++)for(int j=0;j<24;j++){ double s=0; for(int k=0;k<6;k++) s+=B[k][i]*DB[k][j]; Ke[i][j]+=detJ*s; }
    }
}

int main(int argc,char**argv){
    if(argc<3){ fprintf(stderr,"usage: %s m2.mat coordmap.bin [E nu h]\n",argv[0]); return 1; }
    double E=(argc>3)?atof(argv[3]):8.0, nu=(argc>4)?atof(argv[4]):0.49, h=(argc>5)?atof(argv[5]):0.1;
    // ---- load lower-CSC A0 ----
    FILE*f=fopen(argv[1],"rb"); if(!f){perror("mat");return 1;}
    int64_t n64,nnz64; if(fread(&n64,8,1,f)!=1||fread(&nnz64,8,1,f)!=1){return 1;}
    int n=(int)n64; long nnz=(long)nnz64;
    std::vector<long> cs(n+1); std::vector<int> ri(nnz); std::vector<double> va(nnz);
    if(fread(cs.data(),8,n+1,f)!=(size_t)(n+1)||fread(ri.data(),4,nnz,f)!=(size_t)nnz||fread(va.data(),8,nnz,f)!=(size_t)nnz){return 1;}
    fclose(f);
    // ---- coordmap ----
    FILE*g=fopen(argv[2],"rb"); if(!g){perror("coordmap");return 1;}
    int64_t nk64,mt64; if(fread(&nk64,8,1,g)!=1||fread(&mt64,8,1,g)!=1){return 1;}
    long nk=(long)nk64; int mt=(int)mt64;
    std::vector<double> co(3*nk); std::vector<int> na((size_t)mt*nk);
    if(fread(co.data(),8,3*nk,g)!=(size_t)(3*nk)||fread(na.data(),4,(size_t)mt*nk,g)!=(size_t)mt*nk){return 1;}
    fclose(g);
    // disp columns -> eqnode/eqcomp (same logic as gmg)
    std::vector<long> cnt(mt,0);
    for(long nd=0;nd<nk;nd++) for(int j=0;j<mt;j++){int e=na[mt*nd+j]; if(e>=1&&e<=n) cnt[j]++;}
    std::vector<int> ordr(mt); for(int j=0;j<mt;j++) ordr[j]=j;
    std::sort(ordr.begin(),ordr.end(),[&](int a,int c){return cnt[a]>cnt[c];});
    std::vector<int> disp(ordr.begin(),ordr.begin()+3); std::sort(disp.begin(),disp.end());
    std::vector<long> eqnode(n,-1); std::vector<int> eqcomp(n,-1);
    for(long nd=0;nd<nk;nd++) for(int ci=0;ci<3;ci++){int j=disp[ci]; int e=na[mt*nd+j]; if(e>=1&&e<=n){eqnode[e-1]=nd;eqcomp[e-1]=ci;}}
    // eqn index per (node,comp): eqn3[nd*3+comp] = equation (0-based) or -1 (fixed)
    std::vector<int> eqn3((size_t)3*nk,-1);
    for(int e=0;e<n;e++){ long nd=eqnode[e]; int c=eqcomp[e]; if(nd>=0&&c>=0) eqn3[nd*3+c]=e; }
    // lattice ijk per node
    double mn[3]={1e300,1e300,1e300};
    for(long i=0;i<nk;i++) for(int a=0;a<3;a++){ double q=co[3*i+a]; if(q<mn[a])mn[a]=q; }
    std::vector<long> ijk(3*nk);
    for(long i=0;i<nk;i++) for(int a=0;a<3;a++) ijk[3*i+a]=llround((co[3*i+a]-mn[a])/h);
    // node hash: lattice key -> node id
    auto key=[&](long i,long j,long k){ return (i*100000L+j)*100000L+k; };
    std::unordered_map<long,long> nodeat; nodeat.reserve(nk*2);
    for(long nd=0;nd<nk;nd++) nodeat[key(ijk[3*nd],ijk[3*nd+1],ijk[3*nd+2])]=nd;
    // ---- reconstruct elements: a node that is the min-corner of a full 8-node cube ----
    static const int off[8][3]={{0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1}};
    // A0 entry (symmetric lower-CSC) by equation indices (-1 = fixed dof -> 0)
    auto getent=[&](int r,int c)->double{ if(r<0||c<0) return 0.0; if(r<c){int t=r;r=c;c=t;} for(long p=cs[c];p<cs[c+1];p++) if(ri[p]==r) return va[p]; return 0.0; };
    // a REAL element couples its body-diagonal corners (each opposite-corner pair belongs to exactly ONE unit
    // cube); a phantom cube (8 nodes present but no CCX element) has no such coupling in A0 -> reject it.
    static const int diag[4][2]={{0,7},{1,6},{2,5},{3,4}};
    std::vector<std::array<long,8>> elems;
    for(long nd=0;nd<nk;nd++){
        long i=ijk[3*nd],j=ijk[3*nd+1],k=ijk[3*nd+2];
        std::array<long,8> cn; bool full=true;
        for(int c=0;c<8;c++){ auto it=nodeat.find(key(i+off[c][0],j+off[c][1],k+off[c][2])); if(it==nodeat.end()){full=false;break;} cn[c]=it->second; }
        if(!full) continue;
        // reject ONLY if some body-diagonal pair is fully free yet uncoupled in A0 (provably phantom).
        // real elements near the FIXED boundary may have fixed diagonal corners -> keep them.
        bool phantom=false;
        for(int d=0;d<4&&!phantom;d++){ long p=cn[diag[d][0]],q=cn[diag[d][1]];
            bool bothfree = eqn3[p*3]>=0&&eqn3[p*3+1]>=0&&eqn3[p*3+2]>=0
                          &&eqn3[q*3]>=0&&eqn3[q*3+1]>=0&&eqn3[q*3+2]>=0;
            if(!bothfree) continue;
            bool coupled=false;
            for(int a=0;a<3&&!coupled;a++) for(int b=0;b<3;b++) if(getent(eqn3[p*3+a],eqn3[q*3+b])!=0.0){coupled=true;break;}
            if(!coupled) phantom=true;
        }
        if(!phantom) elems.push_back(cn);
    }
    printf("n=%d nnz_lower=%ld nk=%ld elems=%zu\n",n,nnz,nk,elems.size());
    // ---- valence + extraction-feasibility: can we read Ke from valence-1 nodes (one per local corner)? ----
    std::vector<int> valence(nk,0);
    for(auto&cn:elems) for(int c=0;c<8;c++) valence[cn[c]]++;
    { long v1=0; for(long nd=0;nd<nk;nd++) if(valence[nd]==1) v1++;
      int cover[8]={0};
      for(auto&cn:elems){ bool allfree=true;
          for(int c=0;c<8;c++) for(int d=0;d<3;d++) if(eqn3[cn[c]*3+d]<0) allfree=false;
          if(!allfree) continue;
          for(int c=0;c<8;c++) if(valence[cn[c]]==1) cover[c]++; }
      printf("valence-1 nodes=%ld ; clean(all-free-elem) valence-1 sites per local corner: ",v1);
      int ok=0; for(int c=0;c<8;c++){ printf("%d ",cover[c]); if(cover[c]>0) ok++; }
      printf("-> %d/8 corners covered (%s)\n", ok, ok==8?"EXTRACTABLE":"need fallback");
    }
    // ---- Ke: EXTRACT from A0 via valence-1 nodes (exact, formulation-agnostic), or analytic if EXTRACT=0 ----
    double Ke[24][24];
    const char* exenv=getenv("EXTRACT"); int do_extract = !(exenv && atoi(exenv)==0);
    if(do_extract){
        int site[8]; for(int i=0;i<8;i++) site[i]=-1;     // element providing each local corner's row
        for(size_t ei=0;ei<elems.size();ei++){ auto&cn=elems[ei]; bool allfree=true;
            for(int c=0;c<8&&allfree;c++) for(int d=0;d<3;d++) if(eqn3[cn[c]*3+d]<0){allfree=false;break;}
            if(!allfree) continue;
            for(int c=0;c<8;c++) if(valence[cn[c]]==1 && site[c]<0) site[c]=(int)ei; }
        for(int l=0;l<8;l++){ auto&cn=elems[site[l]];
            for(int m=0;m<8;m++) for(int i=0;i<3;i++) for(int j=0;j<3;j++)
                Ke[l*3+i][m*3+j]=getent(eqn3[cn[l]*3+i], eqn3[cn[m]*3+j]); }
        double asym=0; for(int i=0;i<24;i++)for(int j=0;j<24;j++) asym=std::max(asym,std::fabs(Ke[i][j]-Ke[j][i]));
        for(int i=0;i<24;i++)for(int j=i+1;j<24;j++){ double a=0.5*(Ke[i][j]+Ke[j][i]); Ke[i][j]=Ke[j][i]=a; }
        // rigid-body check (Ke @ x-translation ~ 0)
        double tr=0; for(int i=0;i<24;i++){ double s=0; for(int c=0;c<8;c++) s+=Ke[i][3*c]; tr=std::max(tr,std::fabs(s)); }
        printf("EXTRACTED Ke from A0: max asymmetry=%.3e  rigid(x-transl) max|Ke*t|=%.3e\n",asym,tr);
    } else {
        c3d8_Ke(E,nu,h,Ke);
    }
    // ---- random x (free dofs) ----
    std::vector<double> x(n), y_mf(n,0.0), y_as(n,0.0);
    uint64_t s=88172645463325252ull;
    for(int i=0;i<n;i++){ s^=s<<13;s^=s>>7;s^=s<<17; x[i]=(double)((s>>11)&0xFFFFF)/524288.0-1.0; }
    // matrix-free: y += G^T Ke G x  (gather free dofs, 0 for fixed; scatter to free only)
    for(auto&cn:elems){
        double xl[24];
        for(int c=0;c<8;c++) for(int d=0;d<3;d++){ int e=eqn3[cn[c]*3+d]; xl[3*c+d]=(e>=0)?x[e]:0.0; }
        double yl[24];
        for(int i=0;i<24;i++){ double v=0; for(int j=0;j<24;j++) v+=Ke[i][j]*xl[j]; yl[i]=v; }
        for(int c=0;c<8;c++) for(int d=0;d<3;d++){ int e=eqn3[cn[c]*3+d]; if(e>=0) y_mf[e]+=yl[3*c+d]; }
    }
    // assembled symmetric SpMV from lower CSC (diag + sub by column)
    for(int c=0;c<n;c++) for(long p=cs[c];p<cs[c+1];p++){ int r=ri[p]; double v=va[p];
        if(r==c) y_as[c]+=v*x[c]; else { y_as[r]+=v*x[c]; y_as[c]+=v*x[r]; } }
    // compare
    double mxd=0,nrm=0,mxa=0;
    for(int i=0;i<n;i++){ mxd=std::max(mxd,std::fabs(y_mf[i]-y_as[i])); nrm=std::max(nrm,std::fabs(y_as[i])); mxa=std::max(mxa,std::fabs(y_mf[i])); }
    printf("E=%.4g nu=%.4g h=%.4g  ||y_as||inf=%.4e ||y_mf||inf=%.4e  max|y_mf-y_as|=%.4e  rel=%.4e\n",
           E,nu,h,nrm,mxa,mxd,mxd/(nrm+1e-300));
    // debug: where is the error?  worst node + a valence-1 node (must match: only 1 element contributes)
    int worst=0; for(int i=0;i<n;i++) if(std::fabs(y_mf[i]-y_as[i])>std::fabs(y_mf[worst]-y_as[worst])) worst=i;
    printf("worst eqn=%d y_mf=%.4e y_as=%.4e node=%ld comp=%d valence=%d\n",
           worst,y_mf[worst],y_as[worst],eqnode[worst],eqcomp[worst],valence[eqnode[worst]]);
    for(long nd=0;nd<nk;nd++) if(valence[nd]==1){ int e0=eqn3[nd*3+0]; if(e0>=0){
        printf("valence-1 node %ld eqn %d: y_mf=%.4e y_as=%.4e %s\n",nd,e0,y_mf[e0],y_as[e0],
               std::fabs(y_mf[e0]-y_as[e0])<1e-6*std::fabs(y_as[e0]+1e-30)?"MATCH":"DIFFER!"); break; } }
    int ke_ok = (mxd/(nrm+1e-300) < 1e-9);
    printf(ke_ok ? "VERDICT: MATCH (Ke is exact)\n" : "VERDICT: MISMATCH (assembly bug)\n");

    // ================= matrix-free apply: REFERENCE (portable OpenMP) vs ACCEL (Apple AMX dgemm) =========
    // 8-color elements by min-corner lattice parity: same-color cubes are >=2 apart per axis -> share no
    // node -> race-free parallel scatter (no atomics). Standard structured-hex coloring.
    std::vector<std::vector<int>> colored(8);
    for(size_t e=0;e<elems.size();e++){ long nd=elems[e][0];
        int cc=(int)((ijk[3*nd]&1)|((ijk[3*nd+1]&1)<<1)|((ijk[3*nd+2]&1)<<2)); colored[cc].push_back((int)e); }
    double Keflat[576]; for(int i=0;i<24;i++)for(int j=0;j<24;j++) Keflat[i*24+j]=Ke[i][j];   // symmetric -> row==col major
    auto mfapply=[&](const double*xx,double*yy){
        #pragma omp parallel for schedule(static)
        for(int i=0;i<n;i++) yy[i]=0.0;
        for(int c=0;c<8;c++){ auto&E=colored[c]; size_t Nc=E.size(); if(!Nc) continue;
#if CCX_GMG_ACCEL
            std::vector<double> X(24*Nc), Y(24*Nc);
            #pragma omp parallel for schedule(static)
            for(size_t t=0;t<Nc;t++){ auto&cn=elems[E[t]]; for(int a=0;a<8;a++)for(int d=0;d<3;d++){int e=eqn3[cn[a]*3+d]; X[t*24+3*a+d]=(e>=0)?xx[e]:0.0;} }
            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,24,(int)Nc,24,1.0,Keflat,24,X.data(),24,0.0,Y.data(),24);
            #pragma omp parallel for schedule(static)
            for(size_t t=0;t<Nc;t++){ auto&cn=elems[E[t]]; for(int a=0;a<8;a++)for(int d=0;d<3;d++){int e=eqn3[cn[a]*3+d]; if(e>=0) yy[e]+=Y[t*24+3*a+d];} }
#else
            #pragma omp parallel for schedule(static)
            for(size_t t=0;t<Nc;t++){ auto&cn=elems[E[t]]; double xl[24],yl[24];
                for(int a=0;a<8;a++)for(int d=0;d<3;d++){int e=eqn3[cn[a]*3+d]; xl[3*a+d]=(e>=0)?xx[e]:0.0;}
                for(int i=0;i<24;i++){double s=0;for(int j=0;j<24;j++)s+=Keflat[i*24+j]*xl[j];yl[i]=s;}
                for(int a=0;a<8;a++)for(int d=0;d<3;d++){int e=eqn3[cn[a]*3+d]; if(e>=0) yy[e]+=yl[3*a+d];}
            }
#endif
        }
    };
    // assembled baseline: full CSR (symmetric-expanded) + race-free OpenMP row SpMV
    std::vector<int> deg(n,0);
    for(int c=0;c<n;c++) for(long p=cs[c];p<cs[c+1];p++){int r=ri[p]; deg[c]++; if(r!=c) deg[r]++;}
    std::vector<long> rp(n+1,0); for(int i=0;i<n;i++) rp[i+1]=rp[i]+deg[i];
    long fnnz=rp[n]; std::vector<int> fcol(fnnz); std::vector<double> fval(fnnz);
    std::vector<long> curp(rp.begin(),rp.end()-1);
    for(int c=0;c<n;c++) for(long p=cs[c];p<cs[c+1];p++){int r=ri[p];double v=va[p];
        fcol[curp[c]]=r;fval[curp[c]]=v;curp[c]++; if(r!=c){fcol[curp[r]]=c;fval[curp[r]]=v;curp[r]++;}}
    auto csrspmv=[&](const double*xx,double*yy){
        #pragma omp parallel for schedule(static)
        for(int i=0;i<n;i++){ double s=0; for(long p=rp[i];p<rp[i+1];p++) s+=fval[p]*xx[fcol[p]]; yy[i]=s; } };
    std::vector<double> y_csr(n), y_mf2(n);
    csrspmv(x.data(),y_csr.data()); mfapply(x.data(),y_mf2.data());
    double amxd=0,anrm=0,ynorm=0; for(int i=0;i<n;i++){ amxd=std::max(amxd,std::fabs(y_mf2[i]-y_csr[i])); anrm=std::max(anrm,std::fabs(y_csr[i])); ynorm+=y_mf2[i]*y_mf2[i]; }
    ynorm=std::sqrt(ynorm);
    auto tic=[](){return std::chrono::high_resolution_clock::now();};
    auto sec=[](auto a,auto b){return std::chrono::duration<double>(b-a).count();};
    int REPS=30; for(int w=0;w<3;w++){csrspmv(x.data(),y_csr.data()); mfapply(x.data(),y_mf2.data());}
    auto t0=tic(); for(int r=0;r<REPS;r++) csrspmv(x.data(),y_csr.data()); double tcsr=sec(t0,tic())/REPS;
    auto t1=tic(); for(int r=0;r<REPS;r++) mfapply(x.data(),y_mf2.data()); double tmf=sec(t1,tic())/REPS;
    const char* variant = CCX_GMG_ACCEL ? "ACCEL(Apple-AMX-dgemm)" : "REFERENCE(portable-OpenMP)";
    double gb_csr=(double)fnnz*12.0/1e9;
    printf("apply correctness (mf vs CSR): rel=%.3e %s\n", amxd/(anrm+1e-300), amxd/(anrm+1e-300)<1e-9?"OK":"BAD");
    printf("SUMMARY variant=%s ke_ok=%d apply_rel=%.3e elems=%zu csr_apply_s=%.4f mf_apply_s=%.4f speedup=%.2fx csr_GBs=%.1f ynorm=%.10e\n",
           variant, ke_ok, amxd/(anrm+1e-300), elems.size(), tcsr, tmf, tcsr/tmf, gb_csr/tcsr, ynorm);
    return ke_ok ? 0 : 2;
}
