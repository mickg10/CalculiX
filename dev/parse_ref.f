!     parse_ref.f — Fortran REFERENCE field readers, exactly as nodes.f/elements.f do them.
!     The C fast parsers (ccxftoi/ccxftof) must reproduce these bit-for-bit. Tooling / test-only.
!     ftref_i: integer field as read(textpart(1:10),'(i10)')   (the *NODE/*ELEMENT id + kon reads)
!     ftref_f: real    field as read(textpart(1:20),'(f20.0)') (the *NODE coordinate reads)
!     (the stock comma tokenizer reference is splitline.f compiled WITHOUT -DCCX_FAST_PARSE)
      subroutine ftref_i(s, v, istat)
      implicit none
      character*132 s
      integer v, istat
      read(s(1:10),'(i10)',iostat=istat) v
      return
      end
      subroutine ftref_f(s, v, istat)
      implicit none
      character*132 s
      integer istat
      real*8 v
      read(s(1:20),'(f20.0)',iostat=istat) v
      return
      end
