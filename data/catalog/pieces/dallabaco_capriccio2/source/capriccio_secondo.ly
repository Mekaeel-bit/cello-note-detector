\version "2.18.2"

\header {
  title = "[Capriccio Secondo]"
  composer = "Guiseppe Maria Dall' Abaco"
}

capriccio = \context Staff \relative c {

  \time 4/4
  \key g \minor
  \set Staff.midiInstrument = "cello"
  \clef tenor

  \repeat volta 2 {
    \partial 4 r16 d' ees g, |
    fis ees' d a a' d, ees g, fis ees' d a a' d, ees g, | \break % 1
    fis ees' d c d c bes a bes d, g bes a d, d' c | % 2
    bes d, g bes a d, d' c bes a g fis g a bes c \clef bass | \break % 3
    d a d a fis d a fis d4 r16 \clef tenor f'' ees d | % 4
    c bes a bes d, bes' ees, c' f,( bes a8 a16) bes c d | % 5
    ees d c bes a g f ees d d ees f g a bes \clef bass d, | % 6
    c bes' e bes f, a' ees' a, bes, d ees f g a bes d, | % 7
    c bes' e bes f, a' ees' a, bes, f' d' f, a, f' c' f, | % 8
    g, f' bes f ees d ees c' \grace c16 bes4 r16 f, ees f | % 9
    d f bes d f bes d \clef tenor f \grace ees16 d4 c16 bes a bes| % 10
    ees, bes' c bes f a c a \grace c bes4 r16 \clef bass f, ees f | % 11
    d f bes d f bes d f \grace ees d4 c16 bes a bes | % 12
    ees, c' d, bes' ees, c' f, a \grace a bes4 \clef tenor
  }
  r16 d ees g, | % 13
  fis ees' d a a' d, ees g, fis ees' d a a' d, ees g, | % 14
  fis ees' d c d c b c b d f,  aes \clef bass d, f b, d | % 15
  g,8. aes16 g f ees d ees g a! b c ees, d c | % 16
  d a' f' a, d, b' f' b, c, g' a b c ees, d c | % 17
  d a' f' a, d, b' f' b, c,8 c'' r16 \clef tenor ees d f | % 18
  ees d c bes a ees' d f ees d c bes a ees' d f | % 19
  ees d c bes a g fis g fis a d,8 r16 \clef bass g aes c, | % 20
  b aes' d aes g fis g b, c g' ees' d ees g, aes c, | % 21
  b aes' d aes g fis g b, c g' ees' d ees a, bes d, | % 22
  cis bes' e bes a gis a cis, d a' f' e f a, bes d, | % 23
  cis bes' e bes a gis a cis, <c a' dis>4 r16 d' d, d' | % 24
  ees,d' c b c ees, c' ees, d c' bes a bes d, bes' d, | % 25
  c bes' a g a c, a' c, bes d e fis g bes, a g | % 26
  a g' c g d, fis' c' fis, g, d' e fis g bes, a g | % 27
  a g' c g fis e fis a, bes a' d a g fis g bes, | % 28
  c g' ees' d c bes a g fis g fis ees! d c bes a | % 29
  bes d g bes, d, a' g' fis g4 r16 \clef tenor g' fis g | % 30
  d g bes, d \clef bass g, bes d, g bes, d g,8 r16 d g d | % 31
  ees g c, a' a8.\trill g16 g4 r4 | % 32
}

\book {
  \score {
    {
      \clef tenor
      <<
        \new Voice {
          s4 \noBreak s1 \break
          \repeat unfold 5 { s \noBreak s \break }
          s \noBreak s2. \break
          s4 \noBreak s1 \break
          s1 \break
          s1 \noBreak s1 \break \pageBreak
          \repeat unfold 2 { s \noBreak s \break }
          \repeat unfold 7 { s \break }
          \repeat unfold 2 { s \noBreak s \break }
        }
        \new Voice { \capriccio }
      >>
    }
    \layout { indent = 0.0\mm }
    %% uncomment the following line to generate midi.
    %% \midi {}
  }
}
