
#include "header.h"


static const int32_t num_class[] = {  1, };

int32_t get_num_target(void) {
  return N_TARGET;
}
void get_num_class(int32_t* out) {
  for (int i = 0; i < N_TARGET; ++i) {
    out[i] = num_class[i];
  }
}
int32_t get_num_feature(void) {
  return 5;
}
const char* get_threshold_type(void) {
  return "float64";
}
const char* get_leaf_output_type(void) {
  return "float64";
}

void predict(union Entry* data, int pred_margin, double* result) {
  unsigned int tmp;
  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)223.5000000000000284) ) ) {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)168.5000000000000284) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)135.5000000000000284) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)456.5000000000000568) ) ) {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5302.000000000000909) ) ) {
            result[0] += 0.09403072659697631;
          } else {
            result[0] += 0.10860096846119398;
          }
        } else {
          result[0] += 0.03872641238274824;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2839.000000000000455) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)152.5000000000000284) ) ) {
            result[0] += 0.017095235743644258;
          } else {
            result[0] += 0.0729014496101397;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)151.5000000000000284) ) ) {
            result[0] += 0.0862555224354366;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6051.000000000000909) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5140.500000000000909) ) ) {
                result[0] += 0.07459291561554664;
              } else {
                result[0] += 0.10032889969754002;
              }
            } else {
              result[0] += 0.04938403477251238;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)289.5000000000000568) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)276.5000000000000568) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)273.5000000000000568) ) ) {
            result[0] += 0.025171278347360354;
          } else {
            result[0] += 0.0783296559448874;
          }
        } else {
          result[0] += 0.0018659101925157448;
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)9305.000000000001819) ) ) {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4941.000000000000909) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)175.5000000000000284) ) ) {
              result[0] += 0.0697351693360854;
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)479.0000000000000568) ) ) {
                result[0] += 0.0437965036387997;
              } else {
                result[0] += 0.10691655422058038;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)218.5000000000000284) ) ) {
              result[0] += 0.07457313852438521;
            } else {
              result[0] += 0.019966460085109047;
            }
          }
        } else {
          result[0] += -0.008382634197078742;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)261.5000000000000568) ) ) {
      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4989.500000000000909) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)344.5000000000000568) ) ) {
          result[0] += -0.002620220416073328;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)233.5000000000000284) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3668.500000000000455) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2403.500000000000455) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)401.5000000000000568) ) ) {
                  result[0] += -0.00889551725133636;
                } else {
                  result[0] += 0.06165411847811931;
                }
              } else {
                result[0] += 0.10000898314946026;
              }
            } else {
              result[0] += 0.0009560631487040838;
            }
          } else {
            result[0] += 0.01901055603517168;
          }
        }
      } else {
        result[0] += 0.056275643088071176;
      }
    } else {
      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)253.5000000000000284) ) ) {
        result[0] += 0.06912965227564631;
      } else {
        if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)309.5000000000000568) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)283.5000000000000568) ) ) {
            result[0] += -0.01639935699319358;
          } else {
            result[0] += -0.04379590072732976;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)952.0000000000001137) ) ) {
            result[0] += 0.01093985151097291;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)336.5000000000000568) ) ) {
              result[0] += -0.013069792239634302;
            } else {
              result[0] += -0.0348121286383365;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)223.5000000000000284) ) ) {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)168.5000000000000284) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)135.5000000000000284) ) ) {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)982.0000000000001137) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)9103.500000000001819) ) ) {
            result[0] += -0.008357357771694229;
          } else {
            result[0] += 0.04775483826420112;
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)456.5000000000000568) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6805.500000000000909) ) ) {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1656.500000000000227) ) ) {
                result[0] += 0.03748860001021648;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3316.500000000000455) ) ) {
                  result[0] += -0.0047073699157304805;
                } else {
                  result[0] += 0.058298591120512416;
                }
              }
            } else {
              if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)321.5000000000000568) ) ) {
                result[0] += 0.034298217815904496;
              } else {
                result[0] += 0.07341648465058286;
              }
            }
          } else {
            result[0] += -0.008605189021918752;
          }
        }
      } else {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1346.500000000000227) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)998.5000000000001137) ) ) {
            result[0] += -0.028627576635375875;
          } else {
            result[0] += 0.012597338448889462;
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5131.500000000000909) ) ) {
            result[0] += 0.02925896511913749;
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1832.500000000000227) ) ) {
              result[0] += 0.038588436041622626;
            } else {
              result[0] += 0.07432285397765803;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1276.500000000000227) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)451.5000000000000568) ) ) {
          result[0] += -0.030533513809987763;
        } else {
          result[0] += 0.045975567012703625;
        }
      } else {
        if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)289.5000000000000568) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)276.5000000000000568) ) ) {
            result[0] += -0.008853403551940201;
          } else {
            result[0] += -0.04956660248872482;
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5154.000000000000909) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)174.5000000000000284) ) ) {
              result[0] += 0.028491847653950506;
            } else {
              result[0] += 0.0022569800963382334;
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1552.500000000000227) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)416.5000000000000568) ) ) {
                result[0] += 0.02390441082837505;
              } else {
                result[0] += -0.043225357396496536;
              }
            } else {
              result[0] += 0.05592991738900693;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)261.5000000000000568) ) ) {
      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)344.5000000000000568) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5357.500000000000909) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1593.500000000000227) ) ) {
            result[0] += -0.061823744258053204;
          } else {
            result[0] += -0.025722544238373474;
          }
        } else {
          result[0] += 0.004995877791311259;
        }
      } else {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1325.500000000000227) ) ) {
          result[0] += -0.035295997398370985;
        } else {
          result[0] += -0.004274996544755014;
        }
      }
    } else {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)619.5000000000001137) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)324.5000000000000568) ) ) {
          result[0] += -0.05930874159246252;
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)346.5000000000000568) ) ) {
            result[0] += -0.0883024269401953;
          } else {
            result[0] += -0.061977248877647974;
          }
        }
      } else {
        result[0] += 0.030585485642129914;
      }
    }
  }
  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)223.5000000000000284) ) ) {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)168.5000000000000284) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)135.5000000000000284) ) ) {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)982.0000000000001137) ) ) {
          result[0] += 0.011348538270882008;
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)456.5000000000000568) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6805.500000000000909) ) ) {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1656.500000000000227) ) ) {
                result[0] += 0.03569505239537591;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3316.500000000000455) ) ) {
                  result[0] += -0.00447101556660879;
                } else {
                  result[0] += 0.055592716499097417;
                }
              }
            } else {
              result[0] += 0.0643212827429675;
            }
          } else {
            result[0] += -0.008173691888434356;
          }
        }
      } else {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1346.500000000000227) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)998.5000000000001137) ) ) {
            result[0] += -0.027178250693039814;
          } else {
            result[0] += 0.011973334735195794;
          }
        } else {
          result[0] += 0.032500022121516876;
        }
      }
    } else {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1276.500000000000227) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)451.5000000000000568) ) ) {
          result[0] += -0.02899424099144666;
        } else {
          result[0] += 0.043785079537872335;
        }
      } else {
        if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)289.5000000000000568) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)276.5000000000000568) ) ) {
            result[0] += -0.008409734229341144;
          } else {
            result[0] += -0.04711548286434649;
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5305.500000000000909) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)174.5000000000000284) ) ) {
              result[0] += 0.02619278931220764;
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1698.500000000000227) ) ) {
                result[0] += -0.004639267473934355;
              } else {
                result[0] += 0.01114106527073303;
              }
            }
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1444.500000000000227) ) ) {
              result[0] += 0.005690363662438863;
            } else {
              result[0] += 0.04898777933604766;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)261.5000000000000568) ) ) {
      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)344.5000000000000568) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5357.500000000000909) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1593.500000000000227) ) ) {
            result[0] += -0.05879386249915963;
          } else {
            result[0] += -0.02444138048613302;
          }
        } else {
          result[0] += 0.004746774516025234;
        }
      } else {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1268.500000000000227) ) ) {
          result[0] += -0.03641805086371809;
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)248.5000000000000284) ) ) {
            result[0] += 0.003024362780924534;
          } else {
            result[0] += -0.029351457726626992;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)619.5000000000001137) ) ) {
        if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)309.5000000000000568) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1302.500000000000227) ) ) {
            result[0] += -0.08809199309513156;
          } else {
            result[0] += -0.06489282289787128;
          }
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1811.500000000000227) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)952.0000000000001137) ) ) {
              result[0] += -0.03170437756734866;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)292.5000000000000568) ) ) {
                result[0] += -0.05049052162328154;
              } else {
                result[0] += -0.06921275393235912;
              }
            }
          } else {
            result[0] += 0.0036271899869257233;
          }
        }
      } else {
        result[0] += 0.029088501240432347;
      }
    }
  }
  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)223.5000000000000284) ) ) {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)168.5000000000000284) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)135.5000000000000284) ) ) {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)982.0000000000001137) ) ) {
          result[0] += 0.010788638306581853;
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)456.5000000000000568) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5010.500000000000909) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1881.500000000000227) ) ) {
                result[0] += 0.032214222210003085;
              } else {
                result[0] += 0.055858394897773625;
              }
            } else {
              result[0] += 0.05433447881599297;
            }
          } else {
            result[0] += -0.007764164725214017;
          }
        }
      } else {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1346.500000000000227) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)998.5000000000001137) ) ) {
            result[0] += -0.025813196595274698;
          } else {
            result[0] += 0.011380824667329934;
          }
        } else {
          result[0] += 0.030941603487128596;
        }
      }
    } else {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1276.500000000000227) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)451.5000000000000568) ) ) {
          result[0] += -0.02754536274405546;
        } else {
          result[0] += 0.04173590446475109;
        }
      } else {
        if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)289.5000000000000568) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)276.5000000000000568) ) ) {
            result[0] += -0.007988638236799656;
          } else {
            result[0] += -0.044836648202549415;
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5154.000000000000909) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)174.5000000000000284) ) ) {
              result[0] += 0.02582934216641458;
            } else {
              result[0] += 0.0020040300822813536;
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1552.500000000000227) ) ) {
              result[0] += 0.013292351496880178;
            } else {
              result[0] += 0.051349804186428684;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)273.5000000000000568) ) ) {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5357.500000000000909) ) ) {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1372.500000000000227) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2755.500000000000455) ) ) {
            result[0] += -0.037043579734445824;
          } else {
            result[0] += -0.07599624416695959;
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)344.5000000000000568) ) ) {
            result[0] += -0.03445907062927645;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)233.5000000000000284) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3872.500000000000455) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2636.000000000000455) ) ) {
                  result[0] += -0.011356670059256377;
                } else {
                  result[0] += 0.06491970991189729;
                }
              } else {
                result[0] += -0.031034837643672066;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1531.500000000000227) ) ) {
                result[0] += 0.040199964661894315;
              } else {
                result[0] += -0.029885813859755334;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1064.000000000000227) ) ) {
          result[0] += -0.04572607123837499;
        } else {
          result[0] += 0.01867073309422328;
        }
      }
    } else {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)619.5000000000001137) ) ) {
        if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)310.5000000000000568) ) ) {
          result[0] += -0.07715602710934934;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)952.0000000000001137) ) ) {
            result[0] += -0.028030440426254168;
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1729.500000000000227) ) ) {
              result[0] += -0.0630470042947856;
            } else {
              result[0] += -0.00993637820723127;
            }
          }
        }
      } else {
        result[0] += 0.027675574766934143;
      }
    }
  }
  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1552.500000000000227) ) ) {
    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5041.000000000000909) ) ) {
      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)253.5000000000000284) ) ) {
        result[0] += 0.031834410679111226;
      } else {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)958.0000000000001137) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)379.5000000000000568) ) ) {
            result[0] += -0.0645997330020112;
          } else {
            result[0] += -0.036369812908402153;
          }
        } else {
          if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)241.5000000000000284) ) ) {
            result[0] += -0.07221565743134709;
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1325.500000000000227) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)856.5000000000001137) ) ) {
                if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)441.5000000000000568) ) ) {
                  result[0] += 0.04081015969644669;
                } else {
                  result[0] += -0.023720953021455132;
                }
              } else {
                result[0] += -0.035689674832992596;
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4115.500000000000909) ) ) {
                result[0] += -0.025358555131241647;
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1513.500000000000227) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1439.500000000000227) ) ) {
                    result[0] += -0.01748301246648875;
                  } else {
                    result[0] += 0.03019455553367384;
                  }
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1526.500000000000227) ) ) {
                    result[0] += -0.07192472762527306;
                  } else {
                    result[0] += -0.011124947560015586;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1158.500000000000227) ) ) {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)760.0000000000001137) ) ) {
          result[0] += -0.040433447220251655;
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)13082.50000000000182) ) ) {
            result[0] += -0.006277489885538084;
          } else {
            result[0] += 0.05619514729691728;
          }
        }
      } else {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)7068.500000000000909) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1444.500000000000227) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)341.5000000000000568) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)6260.500000000000909) ) ) {
                result[0] += 0.0053730521243195795;
              } else {
                result[0] += 0.04511001607769376;
              }
            } else {
              result[0] += -0.010440247776791138;
            }
          } else {
            result[0] += 0.03009070354572076;
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)454.5000000000000568) ) ) {
            result[0] += 0.0526885805202222;
          } else {
            result[0] += -0.015896496786468286;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5010.500000000000909) ) ) {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1756.500000000000227) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2636.000000000000455) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1630.500000000000227) ) ) {
            result[0] += -0.008272851280823125;
          } else {
            result[0] += -0.04567018555625396;
          }
        } else {
          result[0] += -0.0003573130386146225;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2863.000000000000455) ) ) {
          result[0] += -0.013766237168945629;
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4081.500000000000455) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3842.500000000000455) ) ) {
              result[0] += 0.019283487638985294;
            } else {
              result[0] += -0.02362780600169131;
            }
          } else {
            result[0] += 0.03127779663616436;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5302.000000000000909) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)417.5000000000000568) ) ) {
          result[0] += 0.03947431578753037;
        } else {
          result[0] += -0.008365129547836338;
        }
      } else {
        result[0] += 0.057742654363289195;
      }
    }
  }
  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)210.5000000000000284) ) ) {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)150.5000000000000284) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2917.500000000000455) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)133.5000000000000284) ) ) {
          result[0] += 0.027617366073545285;
        } else {
          result[0] += -0.02383639989386393;
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)126.5000000000000142) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)456.5000000000000568) ) ) {
            if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)258.5000000000000568) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4537.500000000000909) ) ) {
                result[0] += 0.043465191741464876;
              } else {
                result[0] += -0.024869297673408005;
              }
            } else {
              result[0] += 0.04935391016426107;
            }
          } else {
            result[0] += -0.0064797878434401796;
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5010.500000000000909) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4842.500000000000909) ) ) {
              result[0] += 0.0303779085121356;
            } else {
              result[0] += -0.008055417392539314;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5283.500000000000909) ) ) {
              result[0] += 0.060762777709248886;
            } else {
              result[0] += 0.03602539615167697;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)174.5000000000000284) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6051.000000000000909) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)418.5000000000000568) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4901.500000000000909) ) ) {
              result[0] += 0.01968933628609214;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)155.5000000000000284) ) ) {
                result[0] += 0.008178890921800935;
              } else {
                result[0] += 0.05611411414276746;
              }
            }
          } else {
            result[0] += -0.02602313068615854;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6232.500000000000909) ) ) {
            result[0] += -0.05145100023958859;
          } else {
            result[0] += 0.0033787886669986045;
          }
        }
      } else {
        if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)286.5000000000000568) ) ) {
          result[0] += -0.028687406090194812;
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)8204.000000000001819) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5154.000000000000909) ) ) {
              result[0] += 0.0029888493716170106;
            } else {
              result[0] += 0.01916424159566017;
            }
          } else {
            result[0] += -0.0368618478032146;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)249.5000000000000284) ) ) {
      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)347.5000000000000568) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)224.5000000000000284) ) ) {
          result[0] += -0.012685469731326036;
        } else {
          result[0] += -0.03427050709922908;
        }
      } else {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)576.0000000000001137) ) ) {
          result[0] += 0.03650156717029006;
        } else {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4989.500000000000909) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3872.500000000000455) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3418.500000000000455) ) ) {
                result[0] += -0.011084019730621779;
              } else {
                result[0] += 0.0566360356008597;
              }
            } else {
              result[0] += -0.030621338621829336;
            }
          } else {
            result[0] += 0.012130390818059847;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6319.000000000000909) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)317.5000000000000568) ) ) {
          result[0] += -0.04874615427018095;
        } else {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)773.5000000000001137) ) ) {
            result[0] += -0.030628313431674378;
          } else {
            result[0] += -0.07117604734511603;
          }
        }
      } else {
        result[0] += -0.013720089649697183;
      }
    }
  }
  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)210.5000000000000284) ) ) {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)150.5000000000000284) ) ) {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)919.5000000000001137) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)129.5000000000000284) ) ) {
          result[0] += 0.016325913780179047;
        } else {
          result[0] += -0.04192471467144102;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5010.500000000000909) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)129.5000000000000284) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1881.500000000000227) ) ) {
              result[0] += 0.03195631490532935;
            } else {
              result[0] += 0.06203841989701894;
            }
          } else {
            result[0] += 0.02111657886025068;
          }
        } else {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1493.500000000000227) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7542.000000000000909) ) ) {
              result[0] += 0.02055351073437265;
            } else {
              result[0] += 0.0478089137585869;
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)454.5000000000000568) ) ) {
              result[0] += 0.0550289270534107;
            } else {
              result[0] += -0.006219326079154828;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1461.500000000000227) ) ) {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)998.5000000000001137) ) ) {
          result[0] += -0.03227993685147539;
        } else {
          result[0] += -0.004120725974965165;
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5154.000000000000909) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)174.5000000000000284) ) ) {
            result[0] += 0.021014342127518314;
          } else {
            if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)287.5000000000000568) ) ) {
              result[0] += -0.031955104517525194;
            } else {
              result[0] += 0.008784046357439531;
            }
          }
        } else {
          result[0] += 0.037173213470138834;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)249.5000000000000284) ) ) {
      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1534.500000000000227) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5357.500000000000909) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)418.5000000000000568) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4033.500000000000455) ) ) {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1158.500000000000227) ) ) {
                result[0] += -0.0574644583322561;
              } else {
                result[0] += -0.02455874442356973;
              }
            } else {
              result[0] += -0.06434550882140022;
            }
          } else {
            result[0] += 0.01645531274175724;
          }
        } else {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1033.500000000000227) ) ) {
            result[0] += -0.0515261278369804;
          } else {
            result[0] += 0.01070431548250858;
          }
        }
      } else {
        if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)327.5000000000000568) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1920.500000000000227) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2474.000000000000455) ) ) {
              result[0] += 0.025743330144280225;
            } else {
              result[0] += -0.02962893937543304;
            }
          } else {
            result[0] += 0.03819571048470985;
          }
        } else {
          result[0] += 0.005943330911486095;
        }
      }
    } else {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6319.000000000000909) ) ) {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1308.500000000000227) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)764.0000000000001137) ) ) {
            result[0] += -0.03438145104040369;
          } else {
            result[0] += -0.06778352547545939;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)317.5000000000000568) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1699.000000000000227) ) ) {
              result[0] += -0.004482627690454942;
            } else {
              result[0] += -0.04177157923991026;
            }
          } else {
            result[0] += -0.06177521555567396;
          }
        }
      } else {
        result[0] += -0.013044928057409429;
      }
    }
  }
  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)218.5000000000000284) ) ) {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)150.5000000000000284) ) ) {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)919.5000000000001137) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)129.5000000000000284) ) ) {
          result[0] += 0.015530369608675218;
        } else {
          result[0] += -0.03984911643520746;
        }
      } else {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5481.500000000000909) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1873.500000000000227) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1814.500000000000227) ) ) {
              result[0] += 0.029540268057584147;
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)361.5000000000000568) ) ) {
                result[0] += 0.025236508658709786;
              } else {
                result[0] += -0.05378882375984159;
              }
            }
          } else {
            result[0] += 0.04292379832375023;
          }
        } else {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1506.500000000000227) ) ) {
            result[0] += 0.03567757449386087;
          } else {
            result[0] += 0.05804757879704712;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1449.500000000000227) ) ) {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)998.5000000000001137) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)434.5000000000000568) ) ) {
            result[0] += -0.03840827063241132;
          } else {
            result[0] += 0.019434763014956682;
          }
        } else {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2182.500000000000455) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1577.500000000000227) ) ) {
              if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)404.5000000000000568) ) ) {
                result[0] += -0.041553149797665556;
              } else {
                result[0] += 0.023382077027383413;
              }
            } else {
              result[0] += 0.0435910387277515;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)168.5000000000000284) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4322.500000000000909) ) ) {
                result[0] += 0.0294209022947146;
              } else {
                result[0] += -0.007319465144493616;
              }
            } else {
              result[0] += -0.014828866874768205;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4968.500000000000909) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)174.5000000000000284) ) ) {
            result[0] += 0.019299067791949676;
          } else {
            if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)287.5000000000000568) ) ) {
              result[0] += -0.030954702157537012;
            } else {
              result[0] += 0.005870414810884828;
            }
          }
        } else {
          result[0] += 0.036719378531718395;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)261.5000000000000568) ) ) {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1372.500000000000227) ) ) {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5223.500000000000909) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)372.5000000000000568) ) ) {
            result[0] += -0.0565957950703889;
          } else {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1577.500000000000227) ) ) {
              result[0] += 0.0051913321217558666;
            } else {
              result[0] += -0.0640570684269411;
            }
          }
        } else {
          result[0] += -0.008768926297915446;
        }
      } else {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)335.5000000000000568) ) ) {
          result[0] += -0.021429619021552056;
        } else {
          result[0] += -0.0028029901977388145;
        }
      }
    } else {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)619.5000000000001137) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)324.5000000000000568) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1808.500000000000227) ) ) {
            result[0] += -0.047304014537991275;
          } else {
            result[0] += 0.006293796193700739;
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)346.5000000000000568) ) ) {
            result[0] += -0.07122295643919806;
          } else {
            result[0] += -0.04904231359264119;
          }
        }
      } else {
        result[0] += 0.02254233878374631;
      }
    }
  }
  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)220.5000000000000284) ) ) {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)150.5000000000000284) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2917.500000000000455) ) ) {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1330.500000000000227) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)133.5000000000000284) ) ) {
            result[0] += 0.05851841348505063;
          } else {
            result[0] += -0.022933208257427895;
          }
        } else {
          result[0] += -0.04020833350368933;
        }
      } else {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1203.500000000000227) ) ) {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)8479.000000000001819) ) ) {
            result[0] += 0.00409988730445779;
          } else {
            result[0] += 0.04255632202683324;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5081.500000000000909) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)129.5000000000000284) ) ) {
              result[0] += 0.040285928017472945;
            } else {
              result[0] += 0.021739392776153256;
            }
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1493.500000000000227) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1454.500000000000227) ) ) {
                result[0] += 0.039880160869770864;
              } else {
                result[0] += 0.00025008069069670834;
              }
            } else {
              result[0] += 0.04994837320590032;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1449.500000000000227) ) ) {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)998.5000000000001137) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)458.5000000000000568) ) ) {
            result[0] += 0.038184881495144346;
          } else {
            result[0] += -0.03908777337896813;
          }
        } else {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2182.500000000000455) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1577.500000000000227) ) ) {
              result[0] += -0.00274396220923199;
            } else {
              result[0] += 0.04027637841889947;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)168.5000000000000284) ) ) {
              result[0] += 0.006633262011630176;
            } else {
              result[0] += -0.015005581122163651;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5154.000000000000909) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)174.5000000000000284) ) ) {
            result[0] += 0.018644081317838138;
          } else {
            result[0] += -0.00031836475344921173;
          }
        } else {
          result[0] += 0.03468731980751771;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)273.5000000000000568) ) ) {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1325.500000000000227) ) ) {
        result[0] += -0.03759490838868126;
      } else {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)6112.500000000000909) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)240.5000000000000284) ) ) {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1610.500000000000227) ) ) {
              result[0] += -0.02438225007770877;
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3842.500000000000455) ) ) {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3233.500000000000455) ) ) {
                  result[0] += -0.0011971521851674052;
                } else {
                  result[0] += 0.042203541146808486;
                }
              } else {
                result[0] += -0.03567544060537746;
              }
            }
          } else {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3743.500000000000455) ) ) {
              result[0] += -0.03431240120301948;
            } else {
              result[0] += -0.008592340905688845;
            }
          }
        } else {
          result[0] += 0.04820358882235757;
        }
      }
    } else {
      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)773.5000000000001137) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)667.5000000000001137) ) ) {
          result[0] += -0.04243358205792084;
        } else {
          result[0] += 0.021153154932424763;
        }
      } else {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1734.500000000000227) ) ) {
          result[0] += -0.05722893439035102;
        } else {
          result[0] += -0.007417528054087257;
        }
      }
    }
  }
  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)210.5000000000000284) ) ) {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)150.5000000000000284) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2917.500000000000455) ) ) {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1330.500000000000227) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)133.5000000000000284) ) ) {
            result[0] += 0.05593029595179369;
          } else {
            result[0] += -0.021796991013996982;
          }
        } else {
          result[0] += -0.03808083418747717;
        }
      } else {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)919.5000000000001137) ) ) {
          result[0] += -0.0009100960028321664;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)107.5000000000000142) ) ) {
            result[0] += 0.04810488879354434;
          } else {
            result[0] += 0.031635545806356684;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1461.500000000000227) ) ) {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)958.0000000000001137) ) ) {
          result[0] += -0.030105723033215112;
        } else {
          result[0] += -0.003998826036141056;
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5154.000000000000909) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)166.5000000000000284) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)152.5000000000000284) ) ) {
              result[0] += -0.013120673043189274;
            } else {
              result[0] += 0.025958933571706456;
            }
          } else {
            if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)285.5000000000000568) ) ) {
              result[0] += -0.024994916007186195;
            } else {
              result[0] += 0.009484828898790564;
            }
          }
        } else {
          result[0] += 0.03241128942911219;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)249.5000000000000284) ) ) {
      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1534.500000000000227) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5357.500000000000909) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1985.500000000000227) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)392.5000000000000568) ) ) {
              result[0] += -0.02833489485545489;
            } else {
              result[0] += 0.01594131536617541;
            }
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1246.500000000000227) ) ) {
              result[0] += -0.06541039122129665;
            } else {
              result[0] += -0.03442859621377578;
            }
          }
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1426.500000000000227) ) ) {
            result[0] += -0.010818861304238432;
          } else {
            result[0] += 0.04798086603744984;
          }
        }
      } else {
        if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)327.5000000000000568) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2796.500000000000455) ) ) {
            result[0] += 0.01669332669214268;
          } else {
            result[0] += -0.024371494187819764;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)215.5000000000000284) ) ) {
            result[0] += -0.022944850971371653;
          } else {
            result[0] += 0.012216446656151219;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)309.5000000000000568) ) ) {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1630.500000000000227) ) ) {
          result[0] += -0.061344373042845096;
        } else {
          result[0] += -0.015130338257010984;
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7016.000000000000909) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)271.5000000000000568) ) ) {
            result[0] += -0.029314217946282303;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)952.0000000000001137) ) ) {
              result[0] += -0.017599189336491156;
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1773.500000000000227) ) ) {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1075.500000000000227) ) ) {
                  result[0] += -0.07463396770738244;
                } else {
                  result[0] += -0.04763235565394356;
                }
              } else {
                result[0] += 0.010437265930146268;
              }
            }
          }
        } else {
          result[0] += 0.006634447228910376;
        }
      }
    }
  }
  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1552.500000000000227) ) ) {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5608.500000000000909) ) ) {
      if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)263.5000000000000568) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3732.500000000000455) ) ) {
          result[0] += -0.059335852467815836;
        } else {
          result[0] += -0.030093461456650074;
        }
      } else {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1272.500000000000227) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)569.0000000000001137) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)458.5000000000000568) ) ) {
              result[0] += 0.008127462571628958;
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)945.5000000000001137) ) ) {
                result[0] += -0.05155537504637031;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)952.0000000000001137) ) ) {
                  result[0] += 0.003103668223397277;
                } else {
                  result[0] += -0.028809593859234147;
                }
              }
            }
          } else {
            result[0] += 0.0569517931986157;
          }
        } else {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3889.500000000000455) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2775.500000000000455) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2548.500000000000455) ) ) {
                if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)276.5000000000000568) ) ) {
                  result[0] += 0.035515691153171304;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2562.500000000000455) ) ) {
                    if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)337.5000000000000568) ) ) {
                      result[0] += -0.031246034263533132;
                    } else {
                      result[0] += -0.003110516057581042;
                    }
                  } else {
                    result[0] += -0.05569610958943949;
                  }
                }
              } else {
                result[0] += 0.04965241657182327;
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2934.500000000000455) ) ) {
                result[0] += -0.04496184913363414;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3252.500000000000455) ) ) {
                  result[0] += 0.02051676278315745;
                } else {
                  result[0] += -0.0293590336898803;
                }
              }
            }
          } else {
            result[0] += 0.0005052017058908763;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1158.500000000000227) ) ) {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)10743.50000000000182) ) ) {
          result[0] += -0.010665153256264485;
        } else {
          result[0] += 0.026379613640864087;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6319.000000000000909) ) ) {
          result[0] += -0.002615406779918494;
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)456.5000000000000568) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)8337.500000000001819) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)357.5000000000000568) ) ) {
                result[0] += 0.03621444904201536;
              } else {
                result[0] += 0.006100799474426509;
              }
            } else {
              result[0] += 0.05478339567255567;
            }
          } else {
            result[0] += -0.0384162476461438;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5010.500000000000909) ) ) {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1756.500000000000227) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3476.500000000000455) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1567.500000000000227) ) ) {
            result[0] += 0.019474070684323566;
          } else {
            result[0] += -0.01422854449074707;
          }
        } else {
          result[0] += 0.002849726231628559;
        }
      } else {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2528.500000000000455) ) ) {
          result[0] += -0.018145343298026095;
        } else {
          result[0] += 0.016898526741432763;
        }
      }
    } else {
      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5327.500000000000909) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)410.5000000000000568) ) ) {
          result[0] += 0.03486205611935058;
        } else {
          result[0] += -0.004324069256356917;
        }
      } else {
        result[0] += 0.04869469782151095;
      }
    }
  }
  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)212.5000000000000284) ) ) {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)139.5000000000000284) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5010.500000000000909) ) ) {
        result[0] += 0.026887397344728205;
      } else {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1501.500000000000227) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7389.500000000000909) ) ) {
            result[0] += 0.01386350586675103;
          } else {
            result[0] += 0.04096691613027609;
          }
        } else {
          result[0] += 0.04698429093323669;
        }
      }
    } else {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1454.500000000000227) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)168.5000000000000284) ) ) {
          if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)328.5000000000000568) ) ) {
            result[0] += 0.0179385494679586;
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)945.5000000000001137) ) ) {
              result[0] += -0.04983513072200266;
            } else {
              result[0] += -0.0006897790834524364;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2775.500000000000455) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)440.5000000000000568) ) ) {
              result[0] += -0.0030629221050109618;
            } else {
              result[0] += 0.053359205377404476;
            }
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1234.500000000000227) ) ) {
              result[0] += -0.031561144909305124;
            } else {
              result[0] += -0.006694287145412339;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5154.000000000000909) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)173.5000000000000284) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)143.5000000000000284) ) ) {
              if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)311.5000000000000568) ) ) {
                result[0] += 0.034324207601094184;
              } else {
                result[0] += -0.020479110322106888;
              }
            } else {
              result[0] += 0.021387500189197137;
            }
          } else {
            if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)303.5000000000000568) ) ) {
              result[0] += -0.013716425994090002;
            } else {
              result[0] += 0.008403199534164175;
            }
          }
        } else {
          result[0] += 0.0315433763656487;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)248.5000000000000284) ) ) {
      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1756.500000000000227) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)346.5000000000000568) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1064.000000000000227) ) ) {
            result[0] += -0.06636055030018098;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5357.500000000000909) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2986.500000000000455) ) ) {
                result[0] += -0.011068999581195713;
              } else {
                result[0] += -0.03734165057375022;
              }
            } else {
              result[0] += 0.018889024299279657;
            }
          }
        } else {
          result[0] += -0.007436585528128784;
        }
      } else {
        result[0] += 0.011494139919550261;
      }
    } else {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7016.000000000000909) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)952.0000000000001137) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)820.0000000000001137) ) ) {
            result[0] += -0.05312684311385899;
          } else {
            result[0] += -0.0002630099366032154;
          }
        } else {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1325.500000000000227) ) ) {
            result[0] += -0.05837388609258695;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)279.5000000000000568) ) ) {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1534.500000000000227) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)374.5000000000000568) ) ) {
                  result[0] += -0.018765582130322696;
                } else {
                  result[0] += 0.029485708692086273;
                }
              } else {
                result[0] += -0.03959568236842814;
              }
            } else {
              result[0] += -0.04853665939659679;
            }
          }
        }
      } else {
        result[0] += -0.0016466737734328954;
      }
    }
  }
  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)210.5000000000000284) ) ) {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)135.5000000000000284) ) ) {
      if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)258.5000000000000568) ) ) {
        result[0] += 0.0046606143023175915;
      } else {
        result[0] += 0.03535127305172336;
      }
    } else {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1461.500000000000227) ) ) {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)891.5000000000001137) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)192.5000000000000284) ) ) {
            result[0] += -0.039779023455081206;
          } else {
            result[0] += 0.013730671866937822;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)168.5000000000000284) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4533.500000000000909) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3903.500000000000455) ) ) {
                result[0] += 0.010112207086822575;
              } else {
                result[0] += 0.055828209800571654;
              }
            } else {
              result[0] += 0.00025085420857209027;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2606.000000000000455) ) ) {
              result[0] += 0.012258355707573478;
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1234.500000000000227) ) ) {
                result[0] += -0.02954019809011432;
              } else {
                result[0] += -0.006371083250535415;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5154.000000000000909) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)173.5000000000000284) ) ) {
            result[0] += 0.017652560937389927;
          } else {
            if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)287.5000000000000568) ) ) {
              result[0] += -0.02006902539554736;
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1932.500000000000227) ) ) {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1698.500000000000227) ) ) {
                  result[0] += -8.062738150565298e-05;
                } else {
                  result[0] += 0.01962352141968092;
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2009.500000000000227) ) ) {
                  result[0] += -0.044093517465256804;
                } else {
                  result[0] += 0.03359977788367304;
                }
              }
            }
          }
        } else {
          result[0] += 0.03142290162738463;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)248.5000000000000284) ) ) {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1268.500000000000227) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)387.5000000000000568) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6319.000000000000909) ) ) {
            result[0] += -0.054042032514534724;
          } else {
            result[0] += -0.008559229286391535;
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1735.000000000000227) ) ) {
            result[0] += 0.026253659680980052;
          } else {
            result[0] += -0.03537697727544092;
          }
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5357.500000000000909) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1932.500000000000227) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)345.5000000000000568) ) ) {
              result[0] += -0.021409887902394772;
            } else {
              result[0] += -0.0024338832815932205;
            }
          } else {
            result[0] += 0.031417791383936904;
          }
        } else {
          result[0] += 0.022357570422294495;
        }
      }
    } else {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6319.000000000000909) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)952.0000000000001137) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)820.0000000000001137) ) ) {
            result[0] += -0.05104800026527665;
          } else {
            result[0] += -0.00024986917947918304;
          }
        } else {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1288.500000000000227) ) ) {
            result[0] += -0.05829788332468288;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)304.5000000000000568) ) ) {
              result[0] += -0.031323609236150234;
            } else {
              result[0] += -0.05137161378805899;
            }
          }
        }
      } else {
        result[0] += -0.0061221015303061475;
      }
    }
  }
  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)212.5000000000000284) ) ) {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)138.5000000000000284) ) ) {
      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5302.000000000000909) ) ) {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5168.500000000000909) ) ) {
          result[0] += 0.02758945299264008;
        } else {
          result[0] += 0.0022597821150759883;
        }
      } else {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)456.5000000000000568) ) ) {
          result[0] += 0.0406133650961002;
        } else {
          result[0] += -0.0011247199024661163;
        }
      }
    } else {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)174.5000000000000284) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6051.000000000000909) ) ) {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5017.000000000000909) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5283.500000000000909) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1710.500000000000227) ) ) {
                result[0] += -0.013319496635231715;
              } else {
                result[0] += 0.017324250756096626;
              }
            } else {
              result[0] += -0.04421707122810045;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)161.5000000000000284) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5440.500000000000909) ) ) {
                result[0] += 0.05357993753004138;
              } else {
                result[0] += 0.008861148404898473;
              }
            } else {
              result[0] += 0.060513061138894486;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)151.5000000000000284) ) ) {
            result[0] += 0.022425863285103115;
          } else {
            result[0] += -0.007269802124672726;
          }
        }
      } else {
        if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)292.5000000000000568) ) ) {
          result[0] += -0.016328903307890007;
        } else {
          result[0] += 0.0035095085480025823;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)261.5000000000000568) ) ) {
      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4941.000000000000909) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)508.5000000000000568) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)344.5000000000000568) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1842.000000000000227) ) ) {
              result[0] += -0.06707053717252065;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2940.500000000000455) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)222.5000000000000284) ) ) {
                  result[0] += 0.03568034592338406;
                } else {
                  result[0] += -0.021196977016136213;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)258.5000000000000568) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)241.5000000000000284) ) ) {
                    result[0] += -0.025042612080122945;
                  } else {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3702.500000000000455) ) ) {
                      result[0] += -0.07013017690763083;
                    } else {
                      if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)295.5000000000000568) ) ) {
                        result[0] += 0.005989049539485285;
                      } else {
                        result[0] += -0.06034065806469374;
                      }
                    }
                  }
                } else {
                  result[0] += 0.00901359310225806;
                }
              }
            }
          } else {
            result[0] += -0.012787183651009669;
          }
        } else {
          result[0] += 0.047502924727596645;
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)8864.500000000001819) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)218.5000000000000284) ) ) {
            result[0] += 0.048914197168121676;
          } else {
            result[0] += 0.003966443933204327;
          }
        } else {
          result[0] += -0.040245740661846836;
        }
      }
    } else {
      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)689.0000000000001137) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)667.5000000000001137) ) ) {
          result[0] += -0.024953145285278755;
        } else {
          result[0] += 0.044570921754774534;
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)324.5000000000000568) ) ) {
          result[0] += -0.03706670588277101;
        } else {
          result[0] += -0.057060191482507655;
        }
      }
    }
  }
  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)212.5000000000000284) ) ) {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)135.5000000000000284) ) ) {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1539.500000000000227) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)8864.500000000001819) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3116.500000000000455) ) ) {
            result[0] += 0.049367524318133946;
          } else {
            result[0] += 0.014099811996379406;
          }
        } else {
          result[0] += 0.04636616972201808;
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5553.000000000000909) ) ) {
          result[0] += 0.029602766461924987;
        } else {
          result[0] += 0.046607666732712105;
        }
      }
    } else {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1461.500000000000227) ) ) {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)891.5000000000001137) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)197.5000000000000284) ) ) {
            result[0] += -0.03581470697563732;
          } else {
            result[0] += 0.02041110817702191;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1025.500000000000227) ) ) {
            result[0] += 0.05621576613130228;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)168.5000000000000284) ) ) {
              result[0] += 0.00738073506012011;
            } else {
              result[0] += -0.01055132712158963;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5154.000000000000909) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)173.5000000000000284) ) ) {
            result[0] += 0.015966789320737022;
          } else {
            if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)303.5000000000000568) ) ) {
              result[0] += -0.011061437004933374;
            } else {
              result[0] += 0.006976040416407177;
            }
          }
        } else {
          result[0] += 0.02984451884945224;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)248.5000000000000284) ) ) {
      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1756.500000000000227) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5357.500000000000909) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)387.5000000000000568) ) ) {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1246.500000000000227) ) ) {
              result[0] += -0.05535097082466065;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3192.500000000000455) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3084.500000000000455) ) ) {
                  result[0] += -0.011687688515481631;
                } else {
                  result[0] += 0.041169277676518186;
                }
              } else {
                result[0] += -0.030167258778303542;
              }
            }
          } else {
            result[0] += 3.3657447143984484e-05;
          }
        } else {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1033.500000000000227) ) ) {
            result[0] += -0.04392691468632161;
          } else {
            result[0] += 0.015059235999517102;
          }
        }
      } else {
        result[0] += 0.011813427332486299;
      }
    } else {
      if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)309.5000000000000568) ) ) {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1418.500000000000227) ) ) {
          result[0] += -0.058806974404011794;
        } else {
          result[0] += -0.030300146213876267;
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7016.000000000000909) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)952.0000000000001137) ) ) {
            result[0] += -0.005372944986366;
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1075.500000000000227) ) ) {
              result[0] += -0.06109289650852367;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)315.5000000000000568) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1534.500000000000227) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1439.500000000000227) ) ) {
                    result[0] += -0.027421157025905214;
                  } else {
                    result[0] += 0.0008961070946692905;
                  }
                } else {
                  result[0] += -0.040871441377371476;
                }
              } else {
                result[0] += -0.048428858464979754;
              }
            }
          }
        } else {
          result[0] += 0.009796690413660195;
        }
      }
    }
  }
  
  // Apply base_scores
  result[0] += 0;
  
  // Apply postprocessor
  if (!pred_margin) { postprocess(result); }
}

void postprocess(double* result) {
  // sigmoid
  const double alpha = (double)1;
  for (size_t i = 0; i < N_TARGET * MAX_N_CLASS; ++i) {
    result[i] = (double)(1) / ((double)(1) + exp(-alpha * result[i]));
  }
}

