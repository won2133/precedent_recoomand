import { useState, useEffect } from 'react';
import axios from 'axios'
import { useParams } from "react-router";
import Search_box from './search_box.js';
import Menubar from '../elements/menubar';
import Search_title from './search_titlle.js'
import Search_conts from './search_conts.js';
import Paging from '../elements/page.js';
import Header from './header.js';
import LoadingModal from '../elements/LoadingModal';
function Search({apiUrl}) {
  const [searchData, setSearchData] = useState([]);
  const {query, option} = useParams(); //query는 검색 단어, option은 검색 옵션(통합, 민법, 판례)
  const page = 1;
  useEffect(() => {
    async function fetchData() {
      const result = await axios.get(
        apiUrl+"/search?query=" + query + "&option=" + option
      );
       setSearchData(result.data);
    }
    fetchData();
  }, [])
  return (
    <div>
    <header id="header"><Header/></header>
    <nav id="nav">&nbsp;</nav>
    <div id="section">
      <Search_box opt={option} q={query}/>
      {(typeof searchData.a_dic === 'undefined') ? (<LoadingModal/>) : (
          <div id="div" style={{border:'0.1px solid', borderColor: '#DDDDDD', padding: '20px', marginTop:'30px'}}>
             {String(option) !== "3"  && (
                 <div><Search_title name="민법 내용" num={searchData.a_dic.total}/>
                    {searchData.a_dic.total > 0 ?
                         (<Search_conts kind="alist" option={option} list={searchData.a_dic.x_list} word={searchData.a_dic.word} total={searchData.a_dic.total}/>)
                         :(<p>검색 결과가 없습니다.</p>)}
          </div>
        )}
        {String(option) !== "2"  && (
          <div>
             <Search_title name="판례" num={searchData.p_dic.total}/>
                {searchData.p_dic.total > 0 ?
                    (<Search_conts kind="plist" option={option} list={searchData.p_dic.x_list} word={searchData.p_dic.word} total={searchData.p_dic.total}/>)
                    :(<p>검색 결과가 없습니다.</p>)}
          </div>
        )}
      </div>
      )}
    </div>
    <aside id="aside">&nbsp;</aside>
    <footer></footer>
  </div>
    );
};
export default Search;
